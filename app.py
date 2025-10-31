import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageOps
import base64
import io

# --- 1. LOAD THE ONNX MODEL ---
# We load the model into an "Inference Session"
# This is done once when the app starts.
ONNX_MODEL_PATH = 'mnist_model.onnx'
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
print(f"ONNX model loaded from {ONNX_MODEL_PATH}")


# --- 2. IMAGE PROCESSING (Almost Unchanged) ---
# This function is still needed to process the drawing
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


# --- 3. FLASK API SETUP ---
app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('index.html')


# --- 4. UPDATED PREDICT ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'error': 'No image data found'}), 400

    # --- Process the incoming image (same as before) ---
    image_data = data['image_data'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img_rgba = Image.open(io.BytesIO(img_bytes))
    background = Image.new("RGB", img_rgba.size, (255, 255, 255))
    background.paste(img_rgba, (0, 0), img_rgba)
    processed_img = process_image(background)
    if processed_img is None:
        return jsonify({'error': 'No digit drawn'}), 400

    # --- Convert to NumPy (replaces PyTorch transforms) ---
    # 1. Convert PIL image to NumPy array, scale to [0, 1]
    img_array = np.array(processed_img).astype(np.float32) / 255.0

    # 2. Normalize (using the same MNIST stats)
    mean = 0.1307
    std = 0.3081
    img_array = (img_array - mean) / std

    # 3. Add batch and channel dimensions: (28, 28) -> (1, 1, 28, 28)
    img_array = np.expand_dims(img_array, axis=(0, 1))

    # --- Run prediction with ONNX ---
    # We use the input/output names we defined in convert_model.py
    ort_inputs = {'input': img_array}
    ort_outs = ort_session.run(['output'], ort_inputs)

    # ort_outs[0] is the raw output. We find the index with the highest score.
    prediction = np.argmax(ort_outs[0])

    # Send response (as an integer)
    return jsonify({'prediction': int(prediction)})


if __name__ == '__main__':
    # No changes needed here, but remove it for production
    # to avoid confusion. Gunicorn starts the app.
    pass