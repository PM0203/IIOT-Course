# Sense HAT → MQTT → PostgreSQL → Streamlit

This project demonstrates a complete IIoT data pipeline using a **Raspberry Pi Sense HAT** and **open-source tools**.

### 🧠 Overview
1. **publisher.py** – runs on the Raspberry Pi, reads sensor data, and publishes JSON to an MQTT broker.  
2. **server_data_log.py** – subscribes to MQTT and saves messages in daily `.jsonl` log files.  
3. **insert.py** – uploads those logs into a PostgreSQL database (`mqtt_logs` table).  
4. **streamlit_watch.py** – watches the log folder and automatically triggers `insert.py`.  
5. **sense_hat_streamlit_db.py** – a Streamlit dashboard that reads from PostgreSQL and visualizes live data with charts and a 3D orientation model.

### 🚀 Quick Start
1. Install dependencies  
   ```bash
   pip install -r requirements.txt
