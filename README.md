Tick Sightings API — Updated Feature Summary
============================================

This Flask API provides access to tick sighting records stored in a local SQLite database (`tick_sightings.db`). Data can be ingested from the shipped CSV (`TickSightings.csv`) using `parser.py` or uploaded dynamically via the `/upload_csv` endpoint. It supports filtering, analytics, anomaly detection, and optional LSTM-based forecasting.

Core Features
-------------
- SQLite persistence instead of in-memory CSV.
- CSV ingestion: initial load via `parser.py` or runtime upload (`/upload_csv`).
- Filtering by location (match modes), species, and date/time ranges.
- Time-series trends with resampling (daily / weekly / monthly) and moving averages.
- Regional aggregate counts.
- Statistical anomaly detection for recent activity (`/insights/anomalies`).
- Optional LSTM forecast and evaluation endpoints (`/ml/forecast`, `/ml/evaluate`).
- Endpoint index at `/` (returns available endpoints JSON). 404 handler also returns endpoint listings.

Data Model & Storage
--------------------
SQLite table `sightings` columns:
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `date` TEXT (ISO timestamp `YYYY-MM-DDTHH:MM:SS`)
- `location` TEXT
- `species` TEXT
- `latinName` TEXT (optional)

CSV expected columns for ingestion: `date, location, species` (optional `latinName`). Deduplication uses key `(date_iso, lower(location), lower(species))`.

Endpoints Overview
------------------
1. `GET /` — Returns a JSON list of all available endpoints and their HTTP methods.

2. `GET /data` — List raw sighting records.
  Query params (all optional):
  - `location`, `location_match` = contains | prefix | exact
  - `species`
  - `start` / `datetime_start`, `end` / `datetime_end`
  - `limit` (default 100), `offset`
  Example:
  ```bash
  curl "http://127.0.0.1:5000/data?location=Glasgow&start=2018-01-01&end=2019-01-01&limit=10"
  ```

3. `GET /trends` — Time-series counts + moving average.
  Params: same filters as `/data` plus:
  - `interval` = daily|weekly|monthly|D|W|M (default `D`)
  - `window` = moving average size (defaults: 7 for D, 4 for W, 3 for M)
  ```bash
  curl "http://127.0.0.1:5000/trends?location=Nottingham&interval=weekly"
  ```

4. `GET /aggregates/regions` — Count of sightings per location.
  Filters: `location`, `species`, date range, pagination (`limit`, `offset`).
  ```bash
  curl "http://127.0.0.1:5000/aggregates/regions?species=Marsh%20tick&start=2022-01-01&limit=5"
  ```

5. `POST /upload_csv` — Upload and ingest a CSV file.
  - Multipart form field: `file` (CSV)
  - Query param: `mode=replace` to clear existing data before ingest (default `append`).
  Returns ingestion statistics: `rows_total`, `rows_inserted`, `rows_skipped_invalid`, `rows_skipped_duplicate`.
  Example:
  ```bash
  curl -F file=@TickSightings.csv http://127.0.0.1:5000/upload_csv
  ```

6. `GET /insights/anomalies` — Detects unusual activity (z-score > 2) over a recent window.
  - Param: `days` (default 30)
  Returns flagged days with counts and severity.
  ```bash
  curl "http://127.0.0.1:5000/insights/anomalies?days=45"
  ```

7. `GET /ml/forecast` — LSTM-based forecast of future sighting counts.
  - Params: `location`, `species`, `days` (forecast horizon up to 30)
  - Requires a pre-trained model file at `models/lstm_<location or all>_<species or all>.pth`.
  - Response includes predicted counts per future day and metadata.

8. `GET /ml/evaluate` — Returns configuration/details of a trained model for a given location/species combination.

9. 404 — Any unknown path returns JSON with `available_endpoints` for discoverability.

Query & Parsing Behavior
------------------------
- Dates accept `YYYY-MM-DD` (interpreted start-of-day) or `YYYY-MM-DDTHH:MM:SS`; other ISO-like formats parsed by Pandas where possible.
- Location matching: `contains` (default substring), `prefix`, `exact` (case-insensitive logic via LIKE or equality).
- Pagination: `limit` and `offset` applied directly in SQL for performance.
- CSV uploads: duplicates skipped using the composite dedupe key to prevent noisy duplicates.

Initial Data Load
-----------------
Option 1: Run parser to build the database fresh:
```bash
python parser.py  # creates tick_sightings.db from TickSightings.csv
```
Option 2: Start the API and upload a CSV via `/upload_csv`.

Anomaly Detection Details
-------------------------
For each location, daily counts over the last `days` (default 30) are collected. A z-score is computed against that location's mean & std deviation; days with z-score > 2 are flagged (`severity` escalates to `high` above 3).

Forecasting (LSTM)
------------------
The LSTM expects a model file produced by `train_lstm_model` in `ml.py`. Training outline:
1. Aggregate historical daily counts per location/species.
2. Call `train_lstm_model(df, target_column='count', lookback=7, ...)` to produce a `.pth` file.
3. Name the file `models/lstm_<location or all>_<species or all>.pth` so `/ml/forecast` can locate it.
4. Forecast endpoint performs autoregressive prediction for up to `days` future steps (default 14, max 30).

Dependencies
------------
Base (in `requirements.txt`): Flask, pandas, numpy, pytest.
Additional for advanced features:
- `scipy` (installed for statistical operations if needed)
- `torch` (PyTorch for LSTM modeling)
- `scikit-learn` (MinMaxScaler used in `ml.py`)

To install base + extras:
```bash
python -m pip install -r requirements.txt
python -m pip install torch scikit-learn scipy
```

Testing
-------
```bash
python -m pytest
```
Add tests for new endpoints (upload, anomalies, forecast) as needed under `tests/`.

Security & Operational Notes
----------------------------
- File upload is limited to `.csv` by extension check; consider MIME/type validation for production.
- No authentication is currently implemented; restrict network exposure if running with sensitive data.
- Forecasting depends on presence of model files; handle missing models gracefully (already returns 404 with hint).

Future Improvements
-------------------
- Add `/ml/train` endpoint to trigger on-demand model training.
- Implement caching for heavy aggregate queries.
- Add authentication & rate limiting for upload and ML endpoints.
- Integrate more robust anomaly detection (e.g., STL decomposition, seasonal baselines).

License
-------
Internal/informal usage; add a license file if distributing externally.

