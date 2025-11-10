#!/usr/bin/env python3
"""
ingest_file.py

Simple ingestion script that reads a .jsonl file and inserts each record into PostgreSQL.

Usage:
    python ingest_file.py path/to/your_file.jsonl

Environment variables required (set these before running):
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
"""

import os
import sys
import json
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from datetime import datetime

# ---------- CONFIG ----------
STATUS_FILE = Path("upload_status.json")  # for keeping record count summary

# ---------- DATABASE CONNECTION ----------
def pg_connect():
    """Connect to PostgreSQL using environment variables."""
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=os.environ.get("PGPORT", 5432),
        dbname=os.environ.get("PGDATABASE", "postgres"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "admin")
    )
    return conn

# ---------- STATUS LOG ----------
def update_status(file_name: str, record_count: int):
    """Write or update a small JSON status file showing total and last upload count."""
    data = {}
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    total_uploaded = data.get("total_uploaded", 0) + record_count
    data.update({
        "total_uploaded": total_uploaded,
        "last_upload_count": record_count,
        "last_file": file_name,
        "last_update": datetime.now().isoformat(timespec="seconds"),
    })

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[INFO] Updated upload_status.json → total_uploaded={total_uploaded}")

# ---------- DATA INGESTION ----------
def insert_records(conn, file_path: Path):
    """Insert all JSONL records from file into mqtt_logs table."""
    records = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Store bad JSON as raw payload
                data = {"payload": line}

            records.append((
                data.get("received_at"),
                data.get("local_time"),
                data.get("topic"),
                data.get("qos"),
                data.get("retain"),
                data.get("payload"),
                str(file_path.name),
                line_no
            ))

    if not records:
        print(f"[INFO] No valid records found in {file_path}")
        return 0

    insert_sql = """
        INSERT INTO mqtt_logs
        (received_at, local_time, topic, qos, retain, payload, source_file, source_line_no)
        VALUES %s
    """

    with conn.cursor() as cur:
        execute_values(cur, insert_sql, records)
    conn.commit()
    print(f"[INFO] Inserted {len(records)} records from {file_path}")
    return len(records)

# ---------- MAIN ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python ingest_file.py path/to/file.jsonl")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    try:
        conn = pg_connect()
        print(f"[INFO] Connected to PostgreSQL at {os.environ.get('PGHOST', 'localhost')}")
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        sys.exit(1)

    try:
        count = insert_records(conn, file_path)
        update_status(file_path.name, count)
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        conn.rollback()
    finally:
        conn.close()
        print("[INFO] Database connection closed.")

if __name__ == "__main__":
    main()
