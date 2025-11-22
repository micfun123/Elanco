Tick Sightings API — Feature Summary
====================================

This small Flask API provides access to tick sighting records from a CSV file `TickSightings.csv`. It's designed for quick filtering and simple analytics (time series trends and regional aggregates).

Key features
----------------
- Search & filtering by: location, species, and date/time ranges.
- Flexible date parsing: accepts YYYY-MM-DD or ISO datetimes like YYYY-MM-DDTHH:MM:SS.
- Location match modes: contains (default), prefix, or exact.
- Pagination (limit + offset) for results and aggregates.
- Trends endpoint with flexible resampling — daily, weekly, or monthly — and configurable moving average windows.
- Aggregates per region (location) with filtering and pagination.
- Helpful 404 handler that lists all available endpoints & methods.

Endpoints & Usage
----------------------
1) GET /
   - Purpose: List raw sighting records.
   - Query params (all optional):
     - `location` — substring to match location (case-insensitive)
     - `location_match` — one of `contains` (default), `prefix`, `exact`
     - `species` — substring match for species
     - `start` or `datetime_start` — start date (YYYY-MM-DD or ISO), inclusive
     - `end` or `datetime_end` — end date (YYYY-MM-DD or ISO), inclusive
     - `limit` — max number of results to return (integer)
     - `offset` — number of rows to skip (integer)
   - Example: curl "http://127.0.0.1:5000/?location=Glasgow&start=2018-01-01&end=2019-01-01&limit=10"

2) GET /trends
   - Purpose: Return time-series trends — counts per sample interval, plus moving average.
   - Query params:
     - `location` / `location_match` — same behaviour as `/`
     - `species` — filter for species
     - `start` / `datetime_start`, `end` / `datetime_end` — filter by dates
     - `interval` — resample frequency: `daily` (or `D`), `weekly` (`W`), `monthly` (`M`) — default `daily`
     - `window` — moving average window in units of chosen `interval` (defaults to `7` for daily, `4` for weekly, `3` for monthly)
   - Example: curl "http://127.0.0.1:5000/trends?location=Nottingham&interval=weekly"

3) GET /aggregates/regions
   - Purpose: Count of sightings per region (location), with filters
   - Query params:
     - `species` — filter species
     - `location` — filter region (return counts for matching regions only)
     - `start` / `datetime_start`, `end` / `datetime_end` — limit the range before aggregation
     - `limit`, `offset` — pagination for returned regions
   - Example: curl "http://127.0.0.1:5000/aggregates/regions?species=Marsh%20tick&start=2022-01-01&limit=5"

4) 404 responses
   - Unrecognized endpoints return HTTP 404 and a JSON listing available endpoints and methods. That's helpful for discovery and test automation.

How queries & parsing behave
--------------------------------
- Date/time: The API accepts both `YYYY-MM-DD` (interpreted as start-of-day in UTC) and `YYYY-MM-DDTHH:MM:SS` or other parsable ISO datetimes (via Pandas parsing). If a param cannot be parsed, the API returns 400 with an error message.
- Match modes for `location`: 
  - contains (default): a substring search, case-insensitive
  - prefix: location starts with the provided string
  - exact: full match (case-insensitive)
- Pagination: `limit` and `offset` are applied at the end of the `GET /` or `/aggregates/regions` flows.

Running & Testing
----------------------
1) Install dependencies (recommended in a virtualenv):
```bash
python -m pip install -r requirements.txt
```

2) Run the app (development mode):
```bash
python app.py
# or: export FLASK_APP=app.py && flask run
```

3) Run tests with pytest:
```bash
python -m pytest
```

Data source & format
-----------------------
- The app loads `TickSightings.csv` at startup. The CSV must have these columns: 
  `id`, `date` (as ISO string), `location`, `species`, `latinName`.
- The sample data included is already used for default filtering.

Implementation notes
------------------------
- The app kept in `app.py` uses an in-memory CSV read on startup and adds a `date_obj` field (python datetime) for fast comparisons.
- Pandas is used for the time-series resampling and aggregation logic.
- Endpoints return JSON-friendly, consistent result shapes for easy client integration.

