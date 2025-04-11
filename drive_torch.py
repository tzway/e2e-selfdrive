import socketio
import eventlet
import numpy as np
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2
import torch
import torch.nn as nn

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Define Model Architecture (MUST match training exactly)


class NVIDIA_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


sio = socketio.Server()
app = Flask(__name__)
speed_limit = 10


def img_preprocess(img):
    img = img[60:135, :, :]  # Crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img / 255.0
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)

    # Preprocess and convert to tensor
    processed_image = img_preprocess(image)
    tensor_image = torch.tensor(processed_image,
                                dtype=torch.float32).unsqueeze(0).to(device)

    # Predict steering angle
    with torch.no_grad():
        steering_angle = model(tensor_image).item()

    throttle = 1.0 - speed/speed_limit
    print(
        f'Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}')
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Client connected')
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })


if __name__ == '__main__':
    # 2. Create model instance and load weights
    model = NVIDIA_Model().to(device)
    model.load_state_dict(torch.load('torchmodel/ported-torch-model.pth',
                                     map_location=device))
    model.eval()
    print("Model loaded in safe mode")

    # Wrap Flask application
    app = socketio.Middleware(sio, app)

    # Start server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
