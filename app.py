import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageOps
import base64
import io

# --- 1. LOAD THE ONNX MODEL ---
ONNX_MODEL_PATH = 'mnist_model.onnx'
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
print(f"ONNX model loaded from {ONNX_MODEL_PATH}")


# --- 2. IMAGE PROCESSING (Unchanged) ---
# This function takes a black-on-white image,
# inverts it to white-on-black, and centers it.
def process_image(canvas_image):
    # 1. Convert to grayscale AND invert
    img = ImageOps.invert(canvas_image.convert('L'))

    # 2. Find bounding box of the *white* digit
    bbox = img.getbbox()
    if not bbox:
        return None

    # 3. Crop to the digit
    img = img.crop(bbox)

    # 4. Resize to 20x20, preserving aspect ratio
    img.thumbnail((20, 20), Image.LANCZOS)

    # 5. Create a new 28x28 black image (background)
    new_img = Image.new('L', (28, 28), 0)  # 0 = black

    # 6. Paste the 20x20 white digit into the center
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


# --- 4. ***UPDATED AND SIMPLIFIED*** PREDICT ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image_data' not in data:
        return jsonify({'error': 'No image data found'}), 400

    # 1. Decode the Base64 image
    image_data = data['image_data'].split(',')[1]
    img_bytes = base64.b64decode(image_data)

    # 2. Open the image. It's already a black-on-white PNG.
    img = Image.open(io.BytesIO(img_bytes))

    # 3. Process this image directly
    processed_img = process_image(img)
    if processed_img is None:
        return jsonify({'error': 'No digit drawn'}), 400

    # 4. Convert to NumPy (replaces PyTorch transforms)
    img_array = np.array(processed_img).astype(np.float32) / 255.0
    mean = 0.1307
    std = 0.3081
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=(0, 1))  # Shape (1, 1, 28, 28)

    # 5. Run prediction with ONNX
    ort_inputs = {'input': img_array}
    ort_outs = ort_session.run(['output'], ort_inputs)

    prediction = np.argmax(ort_outs[0])

    # 6. Send response
    return jsonify({'prediction': int(prediction)})

# This part is removed as Gunicorn handles running the app
# if __name__ == '__main__':
#    pass