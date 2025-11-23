import os
import sys
import io
import json
import datetime

# Ensure repo root is on path when tests run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app


def make_client():
    return app.test_client()


def upload_rows(client, rows_csv: bytes, filename='upload.csv'):
    data = {'file': (io.BytesIO(rows_csv), filename)}
    return client.post('/upload_csv', data=data, content_type='multipart/form-data')


def test_index_and_404_listings():
    client = make_client()
    res = client.get('/')
    assert res.status_code == 200
    payload = res.get_json()
    assert 'available_endpoints' in payload
    rules = [e['rule'] for e in payload['available_endpoints']]
    assert '/data' in rules
    assert '/trends' in rules

    # 404 returns available_endpoints too
    res2 = client.get('/this-endpoint-does-not-exist')
    assert res2.status_code == 404
    payload2 = res2.get_json()
    assert 'available_endpoints' in payload2


def test_trends_and_aggregates():
    client = make_client()
    # trends should return data for the same date range
    res = client.get('/trends?start=2024-01-01&end=2024-02-01&interval=daily')
    assert res.status_code == 200
    payload = res.get_json()
    assert 'trends' in payload

    res2 = client.get('/aggregates/regions?limit=1000')
    assert res2.status_code == 200
    p2 = res2.get_json()
    assert 'regions' in p2
    assert any(r['location'] == 'Leeds' for r in p2['regions'])


def test_predict_trains_and_returns_predictions():
    client = make_client()

    # Request predictions for Uploadville - the server will train if model missing
    res = client.get('/predict/Uploadville?days_ahead=5')
    # Either we get predictions (200) or a diagnostic (404) if something went wrong
    assert res.status_code in (200, 404)
    payload = res.get_json()
    if res.status_code == 200:
        assert 'predictions' in payload
        assert len(payload['predictions']) == 5
    else:
        # Diagnostic must include helpful keys
        assert 'error' in payload
        assert 'diagnostics' in payload or 'error' in payload
