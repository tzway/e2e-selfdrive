import socketio
import base64
from PIL import Image
import numpy as np
from io import BytesIO
import time

# Create a socket.io client
sio = socketio.Client()

# Connect to your server
sio.connect('http://localhost:4567')

# Generate a dummy image
def generate_dummy_image():
    img = np.ones((160, 320, 3), dtype=np.uint8) * 255  # white image
    pil_img = Image.fromarray(img)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Send a fake telemetry event
def send_fake_telemetry():
    data = {
        "speed": "5.0",
        "image": generate_dummy_image()
    }
    sio.emit("telemetry", data)

# Send 10 fake telemetry frames
for _ in range(10):
    send_fake_telemetry()
    time.sleep(1)

sio.disconnect()
