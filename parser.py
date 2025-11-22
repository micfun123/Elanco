import csv
import sqlite3
import os
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Configuration
CSV_FILE = 'TickSightings.csv'
DB_FILE = 'tick_sightings.db'
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_ONLY_FORMAT = "%Y-%m-%d"

def parse_date(date_raw: str) -> Optional[datetime]:
    """Attempts to parse a date string into a datetime object using multiple formats."""
    if not date_raw:
        return None
    date_raw = date_raw.strip()
    for fmt in (DATE_FORMAT, DATE_ONLY_FORMAT):
        try:
            dt = datetime.strptime(date_raw, fmt)
            if fmt == DATE_ONLY_FORMAT:
                dt = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
            return dt
        except ValueError:
            pass
    try:
        parsed = pd.to_datetime(date_raw)
        if hasattr(parsed, 'to_pydatetime'):
            return parsed.to_pydatetime()
        else:
            return datetime.fromtimestamp(parsed.astype('int64') // 10**9)
    except Exception:
        return None

def convert_csv_to_sql():
    # Remove existing db to start fresh (optional)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            location TEXT NOT NULL,
            species TEXT NOT NULL,
            latinName TEXT
        )
    ''')
    
    # Index for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON sightings (date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_location ON sightings (location)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_species ON sightings (species)')

    print(f"Reading {CSV_FILE}...")
    
    with open(CSV_FILE, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        
        records_to_insert = []
        seen = set()
        
        count_total = 0
        count_valid = 0
        count_dupe = 0

        for row in reader:
            count_total += 1
            
            # extract and strip
            date_raw = (row.get('date') or '').strip()
            location = (row.get('location') or '').strip()
            species = (row.get('species') or '').strip()
            latin = (row.get('latinName') or '').strip() or None

            # Parse Date
            date_obj = parse_date(date_raw)

            if date_obj is None or not location:
                continue

            # Normalization for deduplication
            date_iso = date_obj.strftime(DATE_FORMAT)
            dedupe_key = (date_iso, location.lower(), species.lower())

            if dedupe_key in seen:
                count_dupe += 1
                continue
            
            seen.add(dedupe_key)
            
            # Add to batch
            records_to_insert.append((date_iso, location, species, latin))
            count_valid += 1

    print(f"Total rows: {count_total}")
    print(f"Duplicates skipped: {count_dupe}")
    print(f"Valid rows to insert: {count_valid}")

    # Bulk Insert
    cursor.executemany('INSERT INTO sightings (date, location, species, latinName) VALUES (?, ?, ?, ?)', records_to_insert)
    
    conn.commit()
    conn.close()
    print(f"Successfully created {DB_FILE}")

def ingest_csv_content(raw_bytes: bytes, mode: str = 'append', db_file: str = DB_FILE) -> Dict[str, Any]:
    """Ingest CSV content provided as raw bytes. Returns stats dict.

    mode=replace will clear existing data.
    CSV expected columns: date, location, species (latinName optional).
    Dedup key: (date_iso, lower(location), lower(species)).
    """
    text: str
    try:
        text = raw_bytes.decode('utf-8-sig')
    except Exception:
        return {'error': 'Failed to decode bytes as UTF-8'}

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS sightings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        location TEXT NOT NULL,
        species TEXT NOT NULL,
        latinName TEXT
    )''')
    if mode == 'replace':
        cur.execute('DELETE FROM sightings')

    cur.execute('SELECT date, location, species FROM sightings')
    existing_rows = cur.fetchall()
    existing_keys = {(r['date'], r['location'].lower(), r['species'].lower()) for r in existing_rows}

    reader = csv.DictReader(io.StringIO(text))
    required = {'date', 'location', 'species'}
    if not reader.fieldnames or not required.issubset(set(f.strip() for f in reader.fieldnames if f)):
        conn.close()
        return {'error': f'Missing required columns: {sorted(required)}'}

    total = inserted = skipped_invalid = skipped_duplicate = 0
    batch = []
    for row in reader:
        total += 1
        date_raw = (row.get('date') or '').strip()
        location = (row.get('location') or '').strip()
        species = (row.get('species') or '').strip()
        latin = (row.get('latinName') or '').strip() or None
        dt_obj = parse_date(date_raw)
        if dt_obj is None or not location or not species:
            skipped_invalid += 1
            continue
        date_iso = dt_obj.strftime(DATE_FORMAT)
        key = (date_iso, location.lower(), species.lower())
        if key in existing_keys:
            skipped_duplicate += 1
            continue
        existing_keys.add(key)
        batch.append((date_iso, location, species, latin))
    if batch:
        cur.executemany('INSERT INTO sightings (date, location, species, latinName) VALUES (?, ?, ?, ?)', batch)
        inserted = len(batch)
    conn.commit()
    conn.close()
    return {
        'rows_total': total,
        'rows_inserted': inserted,
        'rows_skipped_invalid': skipped_invalid,
        'rows_skipped_duplicate': skipped_duplicate,
        'mode': mode
    }

import io  # placed at end to avoid circular ordering issues

if __name__ == "__main__":
    convert_csv_to_sql()