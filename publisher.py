#!/usr/bin/env python3
import time
import json
from sense_hat import SenseHat
import paho.mqtt.client as mqtt

sense = SenseHat()

# --- Config ---
BROKER = "broker.hivemq.com"
PORT = 1883
GROUP = 6                # Change to your group number
CLIENTID = f"PublisherGroup{GROUP}"
TOPIC_ALL = f"MSN/group{GROUP}/sensors"   # combined JSON payload
TOPIC_ROLL = f"MSN/group{GROUP}/roll"     # optional individual topics
TOPIC_PITCH = f"MSN/group{GROUP}/pitch"
TOPIC_YAW = f"MSN/group{GROUP}/yaw"
PUBLISH_INTERVAL = 1.0   # seconds

# --- MQTT callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[MQTT] Connected to broker")
    else:
        print("[MQTT] Failed to connect, rc=", rc)

def on_publish(client, userdata, mid):
    # mid is the message id returned by publish()
    print(f"[MQTT] Published message id: {mid}")

# --- Client setup ---
client = mqtt.Client(client_id=CLIENTID)
client.on_connect = on_connect
client.on_publish = on_publish

client.connect(BROKER, PORT, keepalive=60)
client.loop_start()

# --- Main publish loop ---
try:
    while True:
        # Read sensors
        temp = sense.get_temperature()
        hum = sense.get_humidity()
        press = sense.get_pressure()
        o = sense.get_orientation()
        roll = o.get("roll")
        pitch = o.get("pitch")
        yaw = o.get("yaw")

        # Prepare JSON payload
        data = {
            "temperature": round(temp, 2),
            "humidity": round(hum, 2),
            "pressure": round(press, 2),
            "roll": round(roll, 2),
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2),
            "ts": int(time.time() * 1000)   # epoch ms
        }
        payload = json.dumps(data)

        # Publish combined JSON to one topic
        # QoS=0 is fine for lightweight telemetry; change if you need guaranteed delivery
        result = client.publish(TOPIC_ALL, payload, qos=0)
        # result is an MQTTMessageInfo object; its mid will be shown in on_publish
        # If you still want individual topic publishes, uncomment the next lines:

        # client.publish(TOPIC_ROLL, str(round(roll,2)), qos=0)
        # client.publish(TOPIC_PITCH, str(round(pitch,2)), qos=0)
        # client.publish(TOPIC_YAW, str(round(yaw,2)), qos=0)

        print(f"Published to {TOPIC_ALL}: {payload}")

        time.sleep(PUBLISH_INTERVAL)

except KeyboardInterrupt:
    print("Stopped publisher by user (KeyboardInterrupt)")

finally:
    client.loop_stop()
    client.disconnect()
    print("MQTT client disconnected")
