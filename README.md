Tick Sightings API
==================

Lightweight Flask API for managing, querying, and analyzing tick sighting records stored in a local SQLite database (`tick_sightings.db`). It provides CSV ingestion, flexible filters, time-series trends, regional aggregates, anomaly detection, and optional ML forecasting.

Video Demo [https://youtu.be/OtE3PJ5qe7E](https://youtu.be/OtE3PJ5qe7E)

**Quick Start**
- **Install dependencies**:
```bash
python -m pip install -r requirements.txt
```
- **Optional extras (ML & stats)**:
```bash
python -m pip install torch scikit-learn scipy
```
- **Create the DB from supplied CSV (one-time)**:
```bash
Tick Sightings API
==================

A small Flask service for storing, querying and analyzing tick sighting records using SQLite. The app supports CSV ingestion (one-time or runtime upload), flexible filtering, basic time-series trends, regional aggregates, anomaly detection and a simple RandomForest-based forecasting workflow.

Quick Start
-----------
- Install base dependencies:
```bash
python -m pip install -r requirements.txt
```
- Optional ML/statistics extras:
```bash
python -m pip install scikit-learn joblib scipy
```
- Create the SQLite DB from the included CSV (one-time):
```bash
python parser.py    # builds `tick_sightings.db` from `TickSightings.csv`
```
- Run the API:
```bash
python app.py
```

Key files
---------
- `app.py` — main Flask app and endpoints.
- `parser.py` — CSV ingest utilities; `ingest_csv_content` used by `/upload_csv`.
- `train.py` — CLI placeholder to train models (project may include a separate RF trainer).
- `ml.py` — helper utilities (if present) for ML operations.
- `tick_sightings.db` — SQLite DB created by `parser.py` or runtime uploads.

HTTP endpoints (summary)
------------------------
- `GET /` — lists available endpoints and methods.

- `GET /data` — returns raw sighting rows.
  - Filters: `location`, `location_match` (`contains|prefix|exact`), `species`, `start`/`datetime_start`, `end`/`datetime_end`.
  - Pagination: `limit` (default 100), `offset`.

- `GET /trends` — resampled time-series counts plus moving average.
  - Params: `interval` (`daily|weekly|monthly|D|W|M`) and `window` (moving-average size).

- `GET /aggregates/regions` — counts grouped by `location` with pagination.

- `POST /upload_csv` — multipart file upload (`file` field) to ingest a CSV into the DB.
  - Query param: `mode=replace` to clear the table before importing (default `append`).
  - Returns stats: `rows_total`, `rows_inserted`, `rows_skipped_invalid`, `rows_skipped_duplicate`.

- `GET /insights/anomalies` — detect unusual recent activity using z-score.
  - Param: `days` (default 30).

- `GET /predict/<location>/<species>` — forecasts future sighting counts using a RandomForest model.
  - Param: `days_ahead` (default 30).
  - Model file naming: `models/rf_<location>_<species>.pkl`.
  - If a model file is missing the server will attempt to train a simple RandomForestRegressor from historical daily counts (requires at least 14 daily points). If training fails the endpoint returns a diagnostic JSON explaining why.

Dates and parsing
-----------------
- Accepts `YYYY-MM-DD` (start-of-day) or `YYYY-MM-DDTHH:MM:SS`. Pandas parsing is used for other ISO-like formats.
- `location_match` controls matching mode: `contains` (default), `prefix`, or `exact`.

Model training and naming
------------------------
- Models used by `GET /predict/...` are pickled scikit-learn models.
- Naming convention: `models/rf_<location>_<species>.pkl` (spaces replaced by `_` in the code).
- The app will try to train a quick RandomForestRegressor on daily counts when a model is missing. For production or better accuracy, create/train models offline with a richer feature set and save them under `models/`.

Debugging model training
------------------------
- If the predict endpoint returns "Not enough historical data" the response will include diagnostics showing how many daily points were found for exact, LIKE, location-only and species-only queries and small sample rows. Use that to identify mismatched naming or missing data.

Testing
-------
- Run unit tests (recommended):
```bash
python -m pytest -q
```

Troubleshooting
---------------
- If `ModuleNotFoundError: No module named 'app'` appears when running tests directly, run tests with the project root on Python's path:
```bash
PYTHONPATH=$(pwd) python -m pytest -q
```
  or add the repo root to `sys.path` at the top of the test file for quick local runs.

Security & notes
----------------
- The upload endpoint only checks file extension; add MIME/type validation for production.
- No authentication or rate limiting is implemented — avoid exposing the service without access controls.

Future work
-----------
- Add controlled on-demand training endpoints and background jobs.
- Improve forecasting features (more features, forecasting windowing) and model evaluation.
- Harden uploads and add authentication.


***


