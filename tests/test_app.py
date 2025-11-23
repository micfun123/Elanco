import json
from app import app
import io



def test_location_contains():
    client = app.test_client()
    res = client.get('/?location=glas')
    assert res.status_code == 200
    payload = res.get_json()
    assert 'results' in payload
    assert len(payload['results']) > 0


def test_location_exact():
    client = app.test_client()
    res = client.get('/?location=Glasgow&location_match=exact')
    assert res.status_code == 200
    payload = res.get_json()
    assert all(r['location'] == 'Glasgow' for r in payload['results'])


def test_date_range():
    client = app.test_client()
    res = client.get('/?start=2014-01-01&end=2014-12-31')
    assert res.status_code == 200
    payload = res.get_json()
    assert 'results' in payload
    for r in payload['results']:
        assert r['date'].startswith('2014')


def test_pagination():
    client = app.test_client()
    res = client.get('/?limit=2&offset=1')
    assert res.status_code == 200
    payload = res.get_json()
    assert len(payload['results']) <= 2


def test_trends_date_range():
    client = app.test_client()
    res = client.get('/trends?start=2024-01-01&end=2024-12-31')
    assert res.status_code == 200
    payload = res.get_json()
    assert 'trends' in payload
    if payload['trends']:
        for t in payload['trends']:
            assert t['date'].startswith('2024')


def test_404_lists_endpoints():
    client = app.test_client()
    res = client.get('/non-existent-endpoint')
    assert res.status_code == 404
    payload = res.get_json()
    assert 'available_endpoints' in payload
    # We expect at least the index and trends endpoints to be present in the list
    rules = [e['rule'] for e in payload['available_endpoints']]
    assert '/' in rules
    assert '/trends' in rules


def test_aggregates_regions():
    client = app.test_client()
    res = client.get('/aggregates/regions')
    assert res.status_code == 200
    payload = res.get_json()
    assert 'regions' in payload
    assert len(payload['regions']) > 0
    # ensure Glasgow is present
    assert any(r['location'] == 'Glasgow' for r in payload['regions'])


def test_aggregates_regions_with_filters():
    client = app.test_client()
    res = client.get('/aggregates/regions?species=Marsh%20tick&start=2022-01-01&end=2022-12-31')
    assert res.status_code == 200
    payload = res.get_json()
    assert 'regions' in payload
    for r in payload['regions']:
        assert 'count' in r


def test_upload_csv():
    client = app.test_client()
    csv_content = "date,location,species\n2035-01-01T10:00:00,Uploadville,Test tick\n2035-01-02T11:00:00,Uploadville,Test tick".encode('utf-8')
    data = {
        'file': (io.BytesIO(csv_content), 'new_upload.csv')
    }
    res = client.post('/upload_csv', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    payload = res.get_json()
    assert payload['rows_inserted'] == 2
    # Validate that at least one of the new rows is now queryable
    res2 = client.get('/?location=Uploadville&location_match=exact&start=2035-01-01')
    assert res2.status_code == 200
    payload2 = res2.get_json()
    assert len(payload2['results']) >= 2
