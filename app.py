import torch
import torch.nn as nn
import torchvision.transforms as transforms
# --- 1. IMPORT THIS ---
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageOps
import base64
import io


# --- MODEL DEFINITION (Unchanged) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(--1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# --- LOAD MODEL (Unchanged, with optional fix) ---
MODEL_SAVE_PATH = 'mnist_cnn_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
# Added weights_only=True to remove the warning
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
model.eval()
print(f"Model loaded from {MODEL_SAVE_PATH} and running on {device}")


# --- IMAGE PROCESSING (Unchanged) ---
def process_image(canvas_image):
    img = ImageOps.invert(canvas_image.convert('L'))
    bbox = img.getbbox()
    if not bbox:
        return None
    img = img.crop(bbox)
    img.thumbnail((20, 20), Image.LANCZOS)
    new_img = Image.new('L', (28, 28), 0)
    paste_x = (28 - img.width) // 2
    paste_y = (28 - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


# --- FLASK API SETUP ---
app = Flask(__name__)
CORS(app)


# --- 2. ADD THIS NEW ROUTE ---
@app.route('/')
def home():
    # This serves the index.html file from the 'templates' folder
    return render_template('index.html')


# --- PREDICT ROUTE (Unchanged) ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'error': 'No image data found'}), 400

    image_data = data['image_data'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img_rgba = Image.open(io.BytesIO(img_bytes))

    background = Image.new("RGB", img_rgba.size, (255, 255, 255))
    background.paste(img_rgba, (0, 0), img_rgba)

    processed_img = process_image(background)
    if processed_img is None:
        return jsonify({'error': 'No digit drawn'}), 400

    img_tensor = transforms.ToTensor()(processed_img)
    mean = 0.1307
    std = 0.3081
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)

    return jsonify({'prediction': predicted.item()})


if __name__ == '__main__':
    # We still keep this for local testing
    app.run(debug=True, port=5000)