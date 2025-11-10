#!/usr/bin/env python3
import os
import json
import argparse
import threading
import re
from datetime import datetime
import paho.mqtt.client as mqtt

# ----- Configuration & CLI -----
parser = argparse.ArgumentParser(description="MQTT subscriber that batches messages to dated folders/files.")
parser.add_argument("--broker", default="broker.hivemq.com", help="MQTT broker hostname")
parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
parser.add_argument("--group", type=int, default=6, help="Group number for topic")
parser.add_argument("--count", type=int, default=10, help="Number of records per file (batch size)")
parser.add_argument("--outdir", default="logs", help="Root folder where dated folders will be created")
args = parser.parse_args()

BROKER = args.broker
PORT = args.port
GROUP = args.group
TOPIC = f"MSN/group{GROUP}/#"
BATCH_SIZE = max(1, args.count)
ROOT_OUTDIR = args.outdir

# ----- State -----
buffer = []                      # stores incoming records until batch size
buffer_lock = threading.Lock()   # protect buffer from concurrent access
file_counter = -1                 # will be initialized from existing files

# ----- Helpers -----
def date_folder_name(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")

def make_output_folder_for_now() -> str:
    folder = os.path.join(ROOT_OUTDIR, date_folder_name(datetime.now()))
    os.makedirs(folder, exist_ok=True)
    return folder

def unique_filename(prefix="mqtt_batch", ext="jsonl") -> str:
    # Keep it simple per your request: prefix is expected to be numeric (file counter)
    return f"{prefix}.{ext}"

def init_file_counter(out_folder=None):
    """
    Look in today's folder and its 'processed' subfolder for filenames that start
    with a number (e.g., '0.jsonl', '12.jsonl'), find the highest numeric prefix,
    and set the global file_counter to highest + 1.
    If none found, file_counter stays at 0.
    """
    global file_counter

    # ensure we have a valid folder path
    if out_folder is None:
        out_folder = make_output_folder_for_now()

    num_re = re.compile(r"^(\d+)(?:\..*)?$")
    highest = -1

    def scan_dir(path):
        """scan a directory for numeric filenames, return highest number found"""
        local_highest = -1
        try:
            for fname in os.listdir(path):
                m = num_re.match(fname)
                if m:
                    try:
                        n = int(m.group(1))
                        if n > local_highest:
                            local_highest = n
                    except ValueError:
                        continue
        except FileNotFoundError:
            # directory not found, skip
            pass
        except PermissionError:
            pass
        return local_highest

    # 1️⃣ scan the main folder
    main_highest = scan_dir(out_folder)

    # 2️⃣ scan its processed subfolder (if exists)
    processed_folder = os.path.join(out_folder, "processed")
    processed_highest = scan_dir(processed_folder)

    # choose the max across both
    highest = max(main_highest, processed_highest)

    # update global counter
    file_counter = (highest + 1) if highest >= 0 else 0
    print(f"[INFO] Initialized file_counter = {file_counter} (highest existing = {highest})")


def flush_buffer_to_file(force=False):
    """
    Flush current buffer to a new file if it has at least BATCH_SIZE items,
    or if force=True (used on shutdown to save remaining items).
    """
    global buffer, file_counter
    with buffer_lock:
        if not buffer:
            return
        if (len(buffer) >= BATCH_SIZE) or force:
            out_folder = make_output_folder_for_now()
            fname = unique_filename(prefix=f"{file_counter}")
            path = os.path.join(out_folder, fname)
            # Write as JSON Lines (one JSON object per line)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    for record in buffer:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[INFO] Wrote {len(buffer)} records to {path}")
            except Exception as e:
                print(f"[ERROR] Failed writing file {path}: {e}")
            file_counter += 1
            buffer = []

# ----- MQTT callbacks -----
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[INFO] Connected to {BROKER}:{PORT} (rc={rc}). Subscribing to {TOPIC}")
        client.subscribe(TOPIC)
    else:
        print(f"[WARN] Connect returned code {rc}")

def on_message(client, userdata, msg):
    # Prepare a record containing useful metadata + payload (decoded where possible)
    try:
        payload_text = msg.payload.decode("utf-8", errors="replace")
    except Exception:
        payload_text = str(msg.payload)
    record = {
        "received_at": datetime.utcnow().isoformat() + "Z",
        "local_time": datetime.now().isoformat(),
        "topic": msg.topic,
        "qos": msg.qos,
        "retain": bool(msg.retain),
        "payload": payload_text
    }

    with buffer_lock:
        buffer.append(record)
        current_len = len(buffer)

    # If we've reached batch size, flush to file (off the callback lock)
    if current_len >= BATCH_SIZE:
        # flush in a separate thread so callback returns quickly
        threading.Thread(target=flush_buffer_to_file, args=(False,), daemon=True).start()

# ----- Main -----
def main():
    # Initialize file_counter from existing files in today's folder
    init_file_counter()

    client = mqtt.Client(client_id="BatchingSubscriber")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, keepalive=60)
    except Exception as e:
        print(f"[ERROR] Could not connect to broker {BROKER}:{PORT} -> {e}")
        return

    try:
        print("[INFO] Starting loop. Press Ctrl+C to exit and flush remaining messages.")
        client.loop_start()
        # run indefinitely, but let KeyboardInterrupt handle shutdown
        while True:
            # optionally, you could flush periodically if you want time-based flushes
            # here we sleep to reduce CPU usage
            threading.Event().wait(1)
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received. Flushing remaining messages and exiting...")
    finally:
        # Stop network loop, flush remaining messages, and disconnect
        client.loop_stop()
        flush_buffer_to_file(force=True)
        try:
            client.disconnect()
        except Exception:
            pass
        print("[INFO] Exited cleanly.")

if __name__ == "__main__":
    main()
