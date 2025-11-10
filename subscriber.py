# subscriber.py
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883
GROUP = 6
TOPIC = f"MSN/group{GROUP}/#"

def on_connect(client, userdata, flags, rc):
    # rc == 0 means success
    # print(f"Connected (rc={rc})")
    client.subscribe(TOPIC)  # subscribe on (re)connect

def on_message(client, userdata, msg):
    # Print topic, QoS, retain flag, and decoded payload
    print(f"{msg.topic}  QoS={msg.qos}  retain={msg.retain}")
    print("  payload:", msg.payload.decode(errors="replace"))

client = mqtt.Client(client_id="SimpleSubscriber")
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, keepalive=60)
client.loop_forever()  # runs indefinitely