from flask import Flask, request
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Optional
import os


DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_ONLY_FORMAT = "%Y-%m-%d"
csv_path = os.path.join(os.path.dirname(__file__), 'TickSightings.csv')
with open(csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    data = []
    seen = set() 

    for row in reader:
        # Normalize and validate fields
        date_raw = (row.get('date') or '').strip()
        location = (row.get('location') or '').strip()
        species = (row.get('species') or '').strip()

        date_obj = None
        if date_raw:
            try:
                date_obj = datetime.strptime(date_raw, DATE_FORMAT)
            except ValueError:
                try:
                    dt = datetime.strptime(date_raw, DATE_ONLY_FORMAT)
                    date_obj = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
                except ValueError:
                    try:
                        parsed = pd.to_datetime(date_raw)
                        if hasattr(parsed, 'to_pydatetime'):
                            date_obj = parsed.to_pydatetime()
                        else:
                            date_obj = datetime.fromtimestamp(parsed.astype('int64') // 10**9)
                    except Exception:
                        date_obj = None

        if date_obj is None or not location:
            continue

        # Deduplicate based on normalized key: (date_iso, location_lower, species_lower)
        dedupe_key = (date_obj.isoformat(), location.lower(), species.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        cleaned = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        cleaned['date'] = date_obj.strftime(DATE_FORMAT) 
        cleaned['location'] = location
        cleaned['species'] = species
        cleaned['date_obj'] = date_obj

        data.append(cleaned)

app = Flask(__name__)



@app.route('/')
def index():
    location = request.args.get('location')
    datetime_start_str = request.args.get('datetime_start') or request.args.get('start')
    datetime_end_str = request.args.get('datetime_end') or request.args.get('end')
    species = request.args.get('species')

    filtered_data = list(data)

    # Convert filter strings to datetime objects for comparison
    def parse_datetime_param(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        # Try full datetime first
        try:
            return datetime.strptime(s, DATE_FORMAT)
        except ValueError:
            pass
        # Try date only format
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

    datetime_start_obj = parse_datetime_param(datetime_start_str)
    if datetime_start_str and datetime_start_obj is None:
        return {'error': f"Invalid datetime_start format. Use YYYY-MM-DD or {DATE_FORMAT}"}, 400
    datetime_end_obj = parse_datetime_param(datetime_end_str)
    if datetime_end_str and datetime_end_obj is None:
        return {'error': f"Invalid datetime_end format. Use YYYY-MM-DD or {DATE_FORMAT}"}, 400

    # Apply filters
    # Location filter: support match modes: contains (default), exact, prefix
    if location:
        match_mode = request.args.get('location_match', 'contains')
        q = location.strip().lower()
        if match_mode == 'exact':
            filtered_data = [row for row in filtered_data if row.get('location', '').strip().lower() == q]
        elif match_mode == 'prefix':
            filtered_data = [row for row in filtered_data if row.get('location', '').strip().lower().startswith(q)]
        else:
            filtered_data = [row for row in filtered_data if q in row.get('location', '').strip().lower()]

    # Filter by start date
    if datetime_start_obj:
        filtered_data = [
            row for row in filtered_data
            if row['date_obj'] >= datetime_start_obj
        ]

    # Filter by end date
    if datetime_end_obj:
        filtered_data = [
            row for row in filtered_data
            if row['date_obj'] <= datetime_end_obj
        ]

    # Filter by species
    if species:
        q = species.strip().lower()
        filtered_data = [
            row for row in filtered_data
            if q in row.get('species', '').strip().lower()
        ]
    

    # Pagination
    try:
        limit = int(request.args.get('limit')) if request.args.get('limit') else None
    except ValueError:
        return {'error': 'limit must be an integer'}, 400
    try:
        offset = int(request.args.get('offset')) if request.args.get('offset') else 0
    except ValueError:
        return {'error': 'offset must be an integer'}, 400
    results = [
        {key: row[key] for key in row if key != 'date_obj'}
        for row in filtered_data
    ]
    # Apply pagination
    if offset and offset > 0:
        results = results[offset:]
    if limit is not None:
        results = results[:limit]

    return {'results': results}


@app.route('/trends')
def trends():
    location = request.args.get('location')
    species = request.args.get('species')
    filtered_data = list(data)
    datetime_start_str = request.args.get('datetime_start') or request.args.get('start')
    datetime_end_str = request.args.get('datetime_end') or request.args.get('end')
    def parse_datetime_param(s: Optional[str]) -> Optional[datetime]:
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
        

    datetime_start_obj = parse_datetime_param(datetime_start_str)
    if datetime_start_str and datetime_start_obj is None:
        return {'error': f"Invalid datetime_start format. Use YYYY-MM-DD or {DATE_FORMAT}"}, 400
    datetime_end_obj = parse_datetime_param(datetime_end_str)
    if datetime_end_str and datetime_end_obj is None:
        return {'error': f"Invalid datetime_end format. Use YYYY-MM-DD or {DATE_FORMAT}"}, 400
    


    if location:
        match_mode = request.args.get('location_match', 'contains')
        q = location.strip().lower()
        if match_mode == 'exact':
            filtered_data = [row for row in filtered_data if row.get('location', '').strip().lower() == q]
        elif match_mode == 'prefix':
            filtered_data = [row for row in filtered_data if row.get('location', '').strip().lower().startswith(q)]
        else:
            filtered_data = [row for row in filtered_data if q in row.get('location', '').strip().lower()]
    

    if species:
        q = species.strip().lower()
        filtered_data = [
            row for row in filtered_data
            if q in row.get('species', '').strip().lower()
        ]
    
    
    if datetime_start_obj:
        filtered_data = [row for row in filtered_data if row['date_obj'] >= datetime_start_obj]
    if datetime_end_obj:
        filtered_data = [row for row in filtered_data if row['date_obj'] <= datetime_end_obj]

    if not filtered_data:
        return {'error': 'No data available for the given filters.'}, 404
    


    df = pd.DataFrame(filtered_data)
    df['date_obj'] = pd.to_datetime(df['date'], format=DATE_FORMAT)
    df.set_index('date_obj', inplace=True)

    #'interval' for resample frequency: D (day), W (week), M (month). Default D
    interval_str = request.args.get('interval', 'D')
    interval_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'd': 'D', 'w': 'W', 'm': 'M', 'D': 'D', 'W': 'W', 'M': 'M'}
    interval = interval_map.get(interval_str.lower(), 'D')
    counts = df.resample(interval).size()

    # Determine moving average window param (defaults: 7 days, 4 weeks, 3 months)
    try:
        window_param = int(request.args.get('window')) if request.args.get('window') else None
    except ValueError:
        return {'error': 'window must be an integer (number of periods for moving average)'}, 400
    if window_param is None:
        if interval == 'D':
            window_param = 7
        elif interval == 'W':
            window_param = 4
        elif interval == 'M':
            window_param = 3
        else:
            window_param = 7

    moving_avg = counts.rolling(window=window_param).mean()
    trends_data = [
        {
            'date': date.strftime(DATE_FORMAT),
            'sightings_count': int(count),
            'moving_average': float(moving_avg_val) if not np.isnan(moving_avg_val) else None,
            'interval': interval,
            'moving_average_window': int(window_param)
        }
        for date, count, moving_avg_val in zip(counts.index, counts.values, moving_avg.values)
    ]

    return {'trends': trends_data}


@app.route('/aggregates/regions')
def aggregates_by_region():
    """Return number of sightings per region (location). Accepts start/end date and species filters, pagination, and sorting.
    Example: /aggregates/regions?start=2022-01-01&end=2022-12-31&species=Marsh%20tick&limit=10&offset=0
    """
    species = request.args.get('species')
    q_location = request.args.get('location')
    start_str = request.args.get('start') or request.args.get('datetime_start')
    end_str = request.args.get('end') or request.args.get('datetime_end')

    filtered_data = list(data)

    def parse_datetime_param_local(s: Optional[str]) -> Optional[datetime]:
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

    start_obj = parse_datetime_param_local(start_str)
    if start_str and start_obj is None:
        return {'error': f"Invalid start format. Use YYYY-MM-DD or {DATE_FORMAT}"}, 400
    end_obj = parse_datetime_param_local(end_str)
    if end_str and end_obj is None:
        return {'error': f"Invalid end format. Use YYYY-MM-DD or {DATE_FORMAT}"}, 400

    if species:
        q = species.strip().lower()
        filtered_data = [row for row in filtered_data if q in row.get('species', '').strip().lower()]
    if q_location:
        q = q_location.strip().lower()
        filtered_data = [row for row in filtered_data if q in row.get('location', '').strip().lower()]
    if start_obj:
        filtered_data = [row for row in filtered_data if row['date_obj'] >= start_obj]
    if end_obj:
        filtered_data = [row for row in filtered_data if row['date_obj'] <= end_obj]

    if not filtered_data:
        return {'regions': []}

    df = pd.DataFrame(filtered_data)
    counts = df.groupby('location').size().sort_values(ascending=False)

    # Pagination and limit/offset
    try:
        limit = int(request.args.get('limit')) if request.args.get('limit') else None
    except ValueError:
        return {'error': 'limit must be an integer'}, 400
    try:
        offset = int(request.args.get('offset')) if request.args.get('offset') else 0
    except ValueError:
        return {'error': 'offset must be an integer'}, 400

    items = [{'location': loc, 'count': int(count)} for loc, count in counts.items()]
    if offset and offset > 0:
        items = items[offset:]
    if limit is not None:
        items = items[:limit]
    return {'regions': items}



# 404 handler
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
    # Sort endpoints for deterministic output
    endpoints = sorted(endpoints, key=lambda e: (e['rule'], e['endpoint']))
    return {
        'error': 'Not found',
        'available_endpoints': endpoints,
        'usage': 'Use one of the endpoints above, pass query params (e.g., ?location=London) where supported.'
    }, 404


if __name__ == "__main__":
    app.run(debug=True)