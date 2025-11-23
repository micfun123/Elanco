Tick Sightings API
==================

Lightweight Flask API for managing, querying, and analyzing tick sighting records stored in a local SQLite database (`tick_sightings.db`). It provides CSV ingestion, flexible filters, time-series trends, regional aggregates, anomaly detection, and optional ML forecasting.

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
python parser.py 
```
- **Run the API**:
```bash
python app.py
```

**Important Files**
- `app.py` — Flask application and endpoints.
- `parser.py` — CSV ingestion helpers and `ingest_csv_content` used by `/upload_csv`.
- `ml.py` — ML helpers for training and inference (RandomForest-based models).
- `train.py` — Simple CLI to train and save RandomForest model artifacts into `models/`.
- `tick_sightings.db` — SQLite database file created by `parser.py` or via uploads.

**Endpoints (summary & examples)**
- **`GET /`** — lists available endpoints and methods.

- **`GET /data`** — return raw sighting rows.
  - Filters: `location`, `location_match` (`contains|prefix|exact`), `species`, `start`/`datetime_start`, `end`/`datetime_end`.
  - Pagination: `limit` (default 100), `offset`.
  - Example:
```bash
curl "http://127.0.0.1:5000/data?location=Glasgow&start=2018-01-01&end=2019-01-01&limit=10"
```

- **`GET /trends`** — time-series counts and moving average.
  - Additional params: `interval` (`daily|weekly|monthly|D|W|M`), `window` (moving-average window).

- **`GET /aggregates/regions`** — counts grouped by `location`.

- **`POST /upload_csv`** — multipart upload (`file` field) to ingest CSV.
  - Query: `mode=replace` to clear existing data before import (default `append`).
  - Returns ingestion stats: `rows_total`, `rows_inserted`, `rows_skipped_invalid`, `rows_skipped_duplicate`.

- **`GET /insights/anomalies`** — detect unusual recent activity (z-score based).
  - Param: `days` (default 30).

- **`GET /ml/forecast`** and **`GET /ml/evaluate`** — use pre-trained RandomForest models for forecasting/evaluation.
  - Forecast params: `location`, `species`, `days` (max 30).
**Dates & Matching**
- Accepts `YYYY-MM-DD` (start-of-day) or `YYYY-MM-DDTHH:MM:SS`. Pandas parsing is applied for other ISO-like formats.
- `location_match` controls substring/prefix/exact matching.



**Testing**
- Run tests:
```bash
python -m pytest
```

**Operational Notes & Security**
- The upload endpoint enforces `.csv` extension; validate MIME/type in production.
- No authentication included — do not expose the API publicly without safeguards.
- ML endpoints return 404 if no matching model file exists. Train and store a model before requesting forecasts.

**Future Work**
- Add authenticated endpoints, background training tasks, caching for heavy queries, and more robust anomaly detection methods.


