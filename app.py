import sqlite3
import csv
import io
import parser as sightings_parser
from flask import Flask, request, g
from datetime import datetime
import pandas as pd
from typing import Optional
import numpy as np
from scipy import stats
import os
from datetime import timedelta
from ml import load_lstm_model, predict_future

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

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

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
        limit = int(request.args.get('limit', 100)) # Default limit 100
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
    
    # We fetch only the date column for trends to keep it light, 
    # unless you need other columns for complex filtering not handled by SQL.
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

    # Resampling Logic
    interval_str = request.args.get('interval', 'D')
    interval_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'd': 'D', 'w': 'W', 'm': 'M'}
    interval = interval_map.get(interval_str.lower(), 'D')
    
    counts = df.resample(interval).size()

    # Moving Average Logic
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
        # SQLite requires LIMIT if OFFSET is present; use -1 for 'no limit'
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



app.route('/ml/forecast', methods=['GET'])
def lstm_forecast():
    """
    Forecast future tick sightings using trained LSTM.
    
    Query params:
        - location: Filter by location
        - species: Filter by species
        - days: Number of days to forecast (default: 14, max: 30)
    """
    location = request.args.get('location')
    species = request.args.get('species')
    forecast_days = min(int(request.args.get('days', 14)), 30)
    
    try:
        model_path = f'models/lstm_{location or "all"}_{species or "all"}.pth'
        
        if not os.path.exists(model_path):
            return {
                'error': 'Model not found. Train the model first using POST /ml/train',
                'hint': f'Expected model at: {model_path}'
            }, 404
        
        # Load model
        model, scaler, config = load_lstm_model(model_path)
        lookback = config['lookback']
        
        # Get recent data
        query = """
            SELECT date(date) as day, COUNT(*) as count
            FROM sightings
        """
        conditions = []
        params = []
        
        if location:
            conditions.append("location LIKE ?")
            params.append(f'%{location}%')
        if species:
            conditions.append("species LIKE ?")
            params.append(f'%{species}%')
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " GROUP BY day ORDER BY day DESC LIMIT ?"
        params.append(lookback)
        
        df = pd.read_sql_query(query, get_db(), params=params)
        
        if len(df) < lookback:
            return {
                'error': f'Insufficient recent data (have {len(df)} days, need {lookback})'
            }, 400
        
        # Get last sequence (reverse order since we selected DESC)
        last_sequence = df['count'].values[::-1]
        
        # Get last date
        df['day'] = pd.to_datetime(df['day'])
        last_date = df['day'].max()
        
        # Predict future
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions = predict_future(model, scaler, last_sequence, forecast_days, device)
        
        # Format response
        forecast_data = []
        for i, pred_count in enumerate(predictions):
            forecast_date = last_date + timedelta(days=i+1)
            forecast_data.append({
                'date': forecast_date.strftime(DATE_FORMAT),
                'predicted_count': max(0, round(float(pred_count), 2)),
                'confidence': 'high' if i < 7 else 'medium' if i < 14 else 'low'
            })
        
        return {
            'location': location or 'all',
            'species': species or 'all',
            'forecast_days': forecast_days,
            'model_type': 'LSTM',
            'model_config': config,
            'forecast': forecast_data,
            'metadata': {
                'last_actual_date': last_date.strftime(DATE_FORMAT),
                'last_actual_count': int(last_sequence[-1]),
                'model_path': model_path
            }
        }
        
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/ml/evaluate', methods=['GET'])
def evaluate_lstm():
    """Evaluate trained LSTM model performance."""
    location = request.args.get('location')
    species = request.args.get('species')
    
    try:
        model_path = f'models/lstm_{location or "all"}_{species or "all"}.pth'
        
        if not os.path.exists(model_path):
            return {'error': 'Model not found'}, 404
        
        # Load model
        model, scaler, config = load_lstm_model(model_path)
        
        # This would require re-loading and splitting the training data
        # For simplicity, return model info
        return {
            'model_path': model_path,
            'config': config,
            'status': 'Model evaluation would require re-running training data split'
        }
        
    except Exception as e:
        return {'error': str(e)}, 500


if __name__ == "__main__":
    if not os.path.exists(DATABASE):
        print(f"Warning: Database file not found at {DATABASE}")
    app.run(debug=True)