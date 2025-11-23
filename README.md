# Tick Sightings API

Author: Michael Parker
Video Demo [https://youtu.be/OtE3PJ5qe7E](https://youtu.be/OtE3PJ5qe7E)

Lightweight Flask API for storing, querying and analyzing tick sighting records in an on-disk SQLite database (`tick_sightings.db`). This repository provides ingestion from CSV, flexible filtering, time-series trends, regional aggregates, basic anomaly detection, and a simple RandomForest-based forecasting workflow (predictions are by location only).

## Quick start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Build the database from the shipped CSV (one-time):

```bash
python parser.py    # creates tick_sightings.db from TickSightings.csv
```

3. Run the API locally:

```bash
python app.py
```

## Key files

- `app.py` — Flask application exposing the HTTP API.
- `parser.py` — CSV ingestion helpers and `ingest_csv_content` used by `/upload_csv`.


## HTTP endpoints (summary)

- `GET /` — lists available endpoints and methods.

- `GET /data` — list raw sighting rows.
  - Filters: `location`, `location_match` (`contains|prefix|exact`), `species`, `start`/`end`.
  - Pagination: `limit` (default 100), `offset`.

- `GET /trends` — resampled time-series counts plus moving average.
  - Params: `interval` (`daily|weekly|monthly|D|W|M`) and `window` (moving-average window size).

- `GET /aggregates/regions` — counts grouped by `location` (supports filters and pagination).

- `POST /upload_csv` — multipart upload (`file` field) to ingest CSV into the DB.
  - Query param: `mode=replace` to clear existing data before importing (default `append`).
  - Returns ingestion statistics: `rows_total`, `rows_inserted`, `rows_skipped_invalid`, `rows_skipped_duplicate`.

- `GET /insights/anomalies` — z-score based anomaly detection for recent activity.
  - Param: `days` (default 30).

- `GET /predict/<location>` — forecasts future sighting activity for a given `location` (aggregated across species).
  - Query param: `days_ahead` (default 30).
  - Model file naming used by the app: `models/rf_<location>_ALL.pkl`.
  - If a model file is missing the app will attempt to train a simple RandomForestRegressor on historical daily counts for the location (requires at least 14 days of history). For production use, train richer models offline and place them in `models/`.

Example predict request:

```bash
curl "http://127.0.0.1:5000/predict/Leeds?days_ahead=7"
```


## Model training and naming

- Models are scikit-learn RandomForest regressors persisted with `joblib`.
- Auto-trained model naming: `models/rf_<location>_ALL.pkl`.

## Testing

- Run tests with:

```bash
python -m pytest -q
```

If you run tests directly and see `ModuleNotFoundError: No module named 'app'`, run from the repo root or include the project on `PYTHONPATH`:

```bash
PYTHONPATH=$(pwd) python -m pytest -q
```


## Future work

- Add authenticated endpoints, background training jobs, and improved forecasting features.
- Performance & Caching: Cache heavy endpoints (trends/aggregates) in Redis and add materialized views or precomputed aggregates for common time windows.

Tick Sightings API
==================

Lightweight Flask API for managing, querying, and analyzing tick sighting records stored in a local SQLite database (`tick_sightings.db`). It provides CSV ingestion, flexible filters, time-series trends, regional aggregates, anomaly detection, and optional ML forecasting.

Video Demo [https://youtu.be/OtE3PJ5qe7E](https://youtu.be/OtE3PJ5qe7E)
