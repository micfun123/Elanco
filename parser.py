import csv
import sqlite3
import os
import pandas as pd
from datetime import datetime

# Configuration
CSV_FILE = 'TickSightings.csv'
DB_FILE = 'tick_sightings.db'
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
DATE_ONLY_FORMAT = "%Y-%m-%d"

def parse_date(date_raw):
    """Attempts to parse a date string into a datetime object using multiple formats."""
    if not date_raw:
        return None
    
    try:
        return datetime.strptime(date_raw, DATE_FORMAT)
    except ValueError:
        pass

    try:
        dt = datetime.strptime(date_raw, DATE_ONLY_FORMAT)
        return datetime(dt.year, dt.month, dt.day, 0, 0, 0)
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
            species TEXT NOT NULL
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
            records_to_insert.append((date_iso, location, species))
            count_valid += 1

    print(f"Total rows: {count_total}")
    print(f"Duplicates skipped: {count_dupe}")
    print(f"Valid rows to insert: {count_valid}")

    # Bulk Insert
    cursor.executemany('INSERT INTO sightings (date, location, species) VALUES (?, ?, ?)', records_to_insert)
    
    conn.commit()
    conn.close()
    print(f"Successfully created {DB_FILE}")

if __name__ == "__main__":
    convert_csv_to_sql()