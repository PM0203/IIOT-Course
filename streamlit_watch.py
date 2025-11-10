# streamlit_watch_simple.py
import time
import subprocess
import sys
import shutil
import json
from pathlib import Path
import streamlit as st

# ---------- CONFIG ----------
WATCH_FOLDER = "logs/2025-11-03"   # all files live here
POLL_INTERVAL = 30                  # seconds between checks (UI refresh)
AUTO_INGEST_INTERVAL = 30         # seconds (5 minutes) between automatic ingests
INGEST_SCRIPT = Path("streamlit_files/insert.py").resolve()  # adjust path to your ingest script
STATUS_FILE = Path("upload_status.json")         # ingest script may update this
LOGS_DIR = Path.cwd() / "ingest"                 # where per-file ingest logs are written

st.set_page_config(page_title="Log Watcher (Simple)", layout="wide", initial_sidebar_state="collapsed")

# placeholders
badge_ph = st.empty()
notif_ph = st.empty()
action_ph = st.empty()

# session state defaults
if "known_files" not in st.session_state:
    st.session_state.known_files = set()
if "file_count" not in st.session_state:
    st.session_state.file_count = 0
if "last_notified" not in st.session_state:
    st.session_state.last_notified = None
if "last_ingest_time" not in st.session_state:
    st.session_state.last_ingest_time = 0.0
if "ingest_running" not in st.session_state:
    st.session_state.ingest_running = False

watch_path = Path(WATCH_FOLDER)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def current_file_set(p: Path):
    if not p.exists() or not p.is_dir():
        return set()
    return {f.name for f in p.iterdir() if f.is_file()}

def list_pending_files(p: Path):
    """Files in folder that are not already in processed/"""
    if not p.exists() or not p.is_dir():
        return []
    processed_dir = p / "processed"
    processed_names = set()
    if processed_dir.exists() and processed_dir.is_dir():
        processed_names = {f.name for f in processed_dir.iterdir() if f.is_file()}
    pending = [f for f in p.iterdir() if f.is_file() and f.name not in processed_names]
    pending.sort(key=lambda x: x.name)
    return pending

def spawn_ingest_for_file_sync(file_path: Path, log_dir: Path):
    """Run ingest script synchronously and write log; returns (returncode, log_path, err_str)"""
    if not INGEST_SCRIPT.exists():
        return (1, None, f"Ingest script not found at {INGEST_SCRIPT}")
    safe_name = file_path.name.replace(" ", "_")
    log_path = log_dir / f"ingest_{safe_name}.log"
    cmd = [sys.executable, str(INGEST_SCRIPT), str(file_path)]
    with open(log_path, "wb") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=lf)
    return (proc.returncode, log_path, None)

def read_status():
    if not STATUS_FILE.exists():
        return {}
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def run_ingest_cycle(show_ui=True):
    """
    Runs ingestion for all pending files synchronously.
    Returns (succeeded:list, failed:list).
    Updates session_state.last_ingest_time on completion.
    """
    if st.session_state.ingest_running:
        return ([], [("ingest", "already_running")])
    st.session_state.ingest_running = True
    succeeded = []
    failed = []
    try:
        pending = list_pending_files(watch_path)
        if show_ui:
            if not pending:
                st.info("No pending files to ingest.")
                st.session_state.last_ingest_time = time.time()
                return (succeeded, failed)
            else:
                st.info(f"Found {len(pending)} pending file(s). Beginning ingestion...")

        for p in pending:
            if show_ui:
                st.info(f"Ingesting {p.name} ...")
            rc, log_path, err = spawn_ingest_for_file_sync(p, LOGS_DIR)
            if err:
                failed.append((p.name, err))
                continue
            if rc == 0:
                # move to processed/ if still present
                processed_dir = watch_path / "processed"
                processed_dir.mkdir(exist_ok=True)
                if p.exists():
                    dest = processed_dir / p.name
                    if dest.exists():
                        ts = time.strftime("%Y%m%dT%H%M%S")
                        dest = processed_dir / f"{p.stem}_{ts}{p.suffix}"
                    try:
                        shutil.move(str(p), str(dest))
                    except Exception as e:
                        # mark as failed move but continue
                        failed.append((p.name, f"move_failed:{e}"))
                        continue
                succeeded.append((p.name, log_path))
            else:
                failed.append((p.name, f"exit_code:{rc}, log:{log_path}"))

        # update known_files so watcher won't re-notify for moved files
        for name, _ in succeeded:
            st.session_state.known_files.add(name)

        # update last ingest time
        st.session_state.last_ingest_time = time.time()
    finally:
        st.session_state.ingest_running = False

    return (succeeded, failed)

# ---------- initial population ----------
curr_files = current_file_set(watch_path)
if not st.session_state.known_files:
    st.session_state.known_files = curr_files
    st.session_state.file_count = len(curr_files)

# ---------- detect new files ----------
curr_files = current_file_set(watch_path)
new_files = curr_files - st.session_state.known_files

# read status.json for uploaded totals
status = read_status()
uploaded_total = status.get("total_uploaded", 0)
last_upload_count = status.get("last_upload_count", 0)
last_file = status.get("last_file", "N/A")

# update badge (top-right) — shows files seen and uploaded total
badge_html = f"""
<div style="
    position: fixed;
    top: 8px;
    right: 8px;
    background: #0f172a;
    color: #fff;
    padding: 8px 12px;
    border-radius: 10px;
    font-weight: 700;
    z-index: 9999;
    text-align: right;
">
  Files: {st.session_state.file_count:,}<br>
  Uploaded: {uploaded_total:,} recs
</div>
"""
badge_ph.markdown(badge_html, unsafe_allow_html=True)

# single-line pop when new files appear
if new_files:
    st.session_state.file_count += len(new_files)
    st.session_state.known_files.update(new_files)
    st.session_state.last_notified = time.time()
    notif_ph.success("New file encountered")
else:
    notif_ph.empty()

# ---------- manual ingest button ----------
with action_ph.container():
    if st.button("Ingest pending files"):
        succeeded, failed = run_ingest_cycle(show_ui=True)
        if succeeded:
            st.success(f"Ingested {len(succeeded)} file(s).")
        if failed:
            st.error(f"{len(failed)} file(s) failed. See logs for details.")
            for name, reason in failed[:10]:
                st.write(f"- {name}: {reason}")

# ---------- auto-ingest check (runs ingestion if AUTO_INGEST_INTERVAL elapsed) ----------
now = time.time()
time_since_ingest = now - st.session_state.last_ingest_time
if (time_since_ingest >= AUTO_INGEST_INTERVAL) and (not st.session_state.ingest_running):
    # run an automatic ingest but keep UI minimal; show a tiny status message
    st.info("Auto-ingest triggered (5 minutes elapsed since last ingest).")
    succeeded, failed = run_ingest_cycle(show_ui=False)
    # optionally show a quick summary in the notification area
    if succeeded:
        notif_ph.success(f"Auto-ingest: {len(succeeded)} file(s) ingested")
    if failed:
        notif_ph.error(f"Auto-ingest: {len(failed)} files failed (check logs)")

# sleep + rerun (keeps UI updating)
time.sleep(POLL_INTERVAL)
st.rerun()
