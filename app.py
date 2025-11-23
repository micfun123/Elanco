import sqlite3
import csv
import joblib
import parser as sightings_parser
from flask import Flask, request, g
from datetime import datetime
import pandas as pd
from typing import Optional
import numpy as np
from scipy import stats
import os
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

app = Flask(__name__)

# Configuration
DATABASE = os.path.join(os.path.dirname(__file__), 'tick_sightings.db')
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_ONLY_FORMAT = "%Y-%m-%d"

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

def parse_datetime_param(s: Optional[str]) -> Optional[datetime]:
    """Helper to parse input query params into datetime objects."""
    if not s:
        return None
    try:
        return datetime.strptime(s, DATE_FORMAT)
    except ValueError:
        pass
    try:
        dt = datetime.strptime(s, DATE_ONLY_FORMAT)
        return datetime(dt.year, dt.month, dt.day, 0, 0, 0)
    except ValueError:
        pass
    try:
        parsed = pd.to_datetime(s)
        if hasattr(parsed, 'to_pydatetime'):
            return parsed.to_pydatetime()
        return datetime.fromtimestamp(parsed.astype('int64') // 10**9)
    except Exception:
        return None

def build_filters(request_args):
    """Constructs SQL WHERE clauses and parameters based on request args."""
    conditions = []
    params = []

    location = request_args.get('location')
    datetime_start_str = request_args.get('datetime_start') or request_args.get('start')
    datetime_end_str = request_args.get('datetime_end') or request_args.get('end')
    species = request_args.get('species')

    # Date Filters
    start_obj = parse_datetime_param(datetime_start_str)
    if start_obj:
        conditions.append("date >= ?")
        params.append(start_obj.strftime(DATE_FORMAT))

    end_obj = parse_datetime_param(datetime_end_str)
    if end_obj:
        conditions.append("date <= ?")
        params.append(end_obj.strftime(DATE_FORMAT))

    # Species Filter (contains match)
    if species:
        conditions.append("species LIKE ?")
        params.append(f"%{species.strip()}%")

    # Location Filter
    if location:
        match_mode = request_args.get('location_match', 'contains')
        loc_clean = location.strip()
        if match_mode == 'exact':
            conditions.append("location = ?")
            params.append(loc_clean)
        elif match_mode == 'prefix':
            conditions.append("location LIKE ?")
            params.append(f"{loc_clean}%")
        else: # contains
            conditions.append("location LIKE ?")
            params.append(f"%{loc_clean}%")

    return conditions, params


@app.route('/')
def index():
    endpoints = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == 'static':
            continue
        endpoints.append({
            'endpoint': rule.endpoint,
            'methods': sorted([m for m in rule.methods if m not in ['HEAD', 'OPTIONS']]),
            'rule': str(rule)
        })
    return {
        'available_endpoints': sorted(endpoints, key=lambda e: (e['rule'], e['endpoint']))
    }


@app.route('/data')
def data():
    conditions, params = build_filters(request.args)
    
    # Pagination logic
    try:
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
    except ValueError:
        return {'error': 'Limit and offset must be integers'}, 400

    query = "SELECT * FROM sightings"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cur = get_db().execute(query, params)
    results = [dict(row) for row in cur.fetchall()]

    return {'results': results}

@app.route('/trends')
def trends():
    conditions, params = build_filters(request.args)
    query = "SELECT date FROM sightings"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    # Load into Pandas
    conn = get_db()
    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        return {'error': str(e)}, 500

    if df.empty:
        return {'error': 'No data available for the given filters.'}, 404

    # Process Dates
    df['date_obj'] = pd.to_datetime(df['date'])
    df.set_index('date_obj', inplace=True)

    interval_str = request.args.get('interval', 'D')
    interval_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'd': 'D', 'w': 'W', 'm': 'M'}
    interval = interval_map.get(interval_str.lower(), 'D')
    
    counts = df.resample(interval).size()

    # Moving Average
    try:
        window_param = int(request.args.get('window')) if request.args.get('window') else None
    except ValueError:
        return {'error': 'window must be an integer'}, 400

    if window_param is None:
        if interval == 'D': window_param = 7
        elif interval == 'W': window_param = 4
        elif interval == 'M': window_param = 3
        else: window_param = 7

    moving_avg = counts.rolling(window=window_param).mean()

    trends_data = [
        {
            'date': date.strftime(DATE_FORMAT),
            'sightings_count': int(count),
            'moving_average': float(val) if pd.notna(val) else None,
            'interval': interval,
            'moving_average_window': int(window_param)
        }
        for date, count, val in zip(counts.index, counts.values, moving_avg.values)
    ]

    return {'trends': trends_data}

@app.route('/aggregates/regions')
def aggregates_by_region():
    conditions, params = build_filters(request.args)

    # Base Query: Group by location and count
    query = "SELECT location, COUNT(*) as count FROM sightings"
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " GROUP BY location ORDER BY count DESC"

    # Pagination for aggregates
    try:
        limit = int(request.args.get('limit')) if request.args.get('limit') else None
        offset = int(request.args.get('offset')) if request.args.get('offset') else 0
    except ValueError:
        return {'error': 'Limit and offset must be integers'}, 400

    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    elif offset:
        query += " LIMIT -1 OFFSET ?"
        params.append(offset)

    cur = get_db().execute(query, params)
    items = [dict(row) for row in cur.fetchall()]

    return {'regions': items}


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Upload a CSV file and ingest new sighting rows.
    Expected columns: date, location, species (latinName optional).
    Query params:
      mode=replace   -> clears existing data before ingest
    Response includes counts of processed, inserted, duplicate/invalid skipped.
    """
    file = request.files.get('file')
    if file is None or file.filename == '':
        return {'error': 'No file provided'}, 400
    if not file.filename.lower().endswith('.csv'):
        return {'error': 'File must be a .csv'}, 400

    mode = request.args.get('mode', 'append').lower()
    try:
        raw_content = file.read()
    except Exception as e:
        return {'error': f'Failed to read file: {e}'}, 400
    try:
        text = raw_content.decode('utf-8-sig')
    except Exception:
        return {'error': 'Failed to decode file as UTF-8'}, 400

    stats = sightings_parser.ingest_csv_content(raw_content, mode=mode, db_file=DATABASE)
    if 'error' in stats:
        return stats, 400
    return stats


@app.route('/insights/anomalies')
def detect_anomalies():
    """
    Detect unusual tick activity using statistical methods.
    Flags locations with sightings significantly above normal.
    """
    days_back = int(request.args.get('days', 30))
    
    # Get daily counts by location
    query = """
        SELECT 
            location,
            date(date) as day,
            COUNT(*) as daily_count
        FROM sightings
        WHERE date >= date('now', '-' || ? || ' days')
        GROUP BY location, day
    """
    
    df = pd.read_sql_query(query, get_db(), params=[days_back])
    
    anomalies = []
    
    # For each location, detect outliers using z-score
    for location in df['location'].unique():
        loc_data = df[df['location'] == location]
        counts = loc_data['daily_count'].values
        
        if len(counts) < 7:  # Need minimum data
            continue
        
        mean = np.mean(counts)
        std = np.std(counts)
        
        # Find days with z-score > 2 (unusual)
        for _, row in loc_data.iterrows():
            z_score = (row['daily_count'] - mean) / (std + 0.001)
            
            if z_score > 2:  # Significantly above average
                anomalies.append({
                    'location': location,
                    'date': row['day'],
                    'sighting_count': int(row['daily_count']),
                    'expected_count': round(mean, 1),
                    'severity': 'high' if z_score > 3 else 'medium',
                    'z_score': round(z_score, 2)
                })
    
    return {
        'anomalies_detected': len(anomalies),
        'analysis_period_days': days_back,
        'alerts': sorted(anomalies, key=lambda x: x['z_score'], reverse=True)
    }

def train_rf_model(location: str, model_path: str):
    """
    Trains a RandomForestRegressor on historical daily counts for a LOCATION only.
    Aggregates all species together.
    """
    loc = (location or '').strip()
    MIN_POINTS = 14

    conn = get_db()
    
    try:
        # Exact match on location
        q_exact = """
            SELECT date(date) as day, COUNT(*) as cnt
            FROM sightings
            WHERE LOWER(location) = LOWER(?)
            GROUP BY day
            ORDER BY day
        """
        df = pd.read_sql_query(q_exact, conn, params=[loc])
        
        # Fallback to LIKE match if exact is too sparse
        if df.empty or len(df) < MIN_POINTS:
            q_like = """
                SELECT date(date) as day, COUNT(*) as cnt
                FROM sightings
                WHERE location LIKE ?
                GROUP BY day
                ORDER BY day
            """
            df_like = pd.read_sql_query(q_like, conn, params=[f"%{loc}%"])
            if len(df_like) >= MIN_POINTS:
                df = df_like
    except Exception as e:
        return {'error': f'Failed to read historical data: {e}'}

    if df.empty or len(df) < MIN_POINTS:
        return {'error': 'Not enough historical data to train model (need >= 14 days of activity)'}

   
    df['day_dt'] = pd.to_datetime(df['day'])
    df = df.set_index('day_dt').sort_index()
    

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_range, fill_value=0)
    
    # 7-day rolling average
    df['rolling_avg'] = df['cnt'].rolling(window=7, min_periods=1).mean()
    
    df['doy'] = df.index.dayofyear
    X = df[['doy']].values
    y = df['rolling_avg'].values

    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, model_path)
        return {'trained': True, 'rows_used': int(len(df)), 'model_path': model_path}
    except Exception as e:
        return {'error': f'Failed to train/save model: {e}'}


@app.route('/predict/<location>')
def predict_sightings(location):
    """
    Predicts aggregate tick activity for a location (ignoring species).
    """
    days_ahead = int(request.args.get('days_ahead', 30))
    model_path = f"models/rf_{location.replace(' ', '_')}_ALL.pkl"
    
    if not os.path.exists(model_path):
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            train_result = train_rf_model(location, model_path)
            if 'error' in train_result:
                return train_result, 404
        except Exception as e:
            return {'error': f'Failed to train model: {e}'}, 500

    try:
        model = joblib.load(model_path)
    except Exception:
        train_rf_model(location, model_path)
        model = joblib.load(model_path)

    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future_features = np.array([[d.timetuple().tm_yday] for d in future_dates])
    predictions = model.predict(future_features)

    results = [
        {
            'date': d.strftime(DATE_ONLY_FORMAT),
            'predicted_sightings': round(float(pred), 2)
        }
        for d, pred in zip(future_dates, predictions)
    ]

    return {
        'location': location,
        'scope': 'All Species',
        'model_type': 'RandomForest (7-day Rolling Avg)',
        'predictions': results
    }
   

@app.errorhandler(404)
def not_found(error):
    endpoints = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint == 'static':
            continue
        endpoints.append({
            'endpoint': rule.endpoint,
            'methods': sorted([m for m in rule.methods if m not in ['HEAD', 'OPTIONS']]),
            'rule': str(rule)
        })
    return {
        'error': 'Not found',
        'available_endpoints': sorted(endpoints, key=lambda e: (e['rule'], e['endpoint']))
    }, 404
  

if __name__ == "__main__":
    if not os.path.exists(DATABASE):
        print(f"Warning: Database file not found at {DATABASE}")
    app.run(debug=True)