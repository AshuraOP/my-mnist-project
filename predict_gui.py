# --- This is the full, corrected code for 'predict_gui.py' ---

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import tkinter as tk
from tkinter import font
import io
# We need ImageDraw to draw on our in-memory PIL image
from PIL import Image, ImageOps, ImageDraw
import numpy as np


# --- 1. DEFINE THE MODEL ARCHITECTURE ---
# (This section is unchanged)
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
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. LOAD THE TRAINED MODEL ---
# (This section is unchanged)
MODEL_SAVE_PATH = 'mnist_cnn_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_SAVE_PATH} and running on {device}")


# --- 3. IMAGE PROCESSING FUNCTION ---
# --- THIS IS THE CORRECTED SECTION ---
def process_image(canvas_image):
    # 1. Convert to grayscale AND invert
    # We draw in black, but MNIST needs a white digit.
    # So, we invert the image right at the start.
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
    # The backgrounds (both black) will match perfectly.
    paste_x = (28 - img.width) // 2
    paste_y = (28 - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))

    # 7. No final invert needed. The image is now correct.
    return new_img


# --- END OF CORRECTED SECTION ---


# --- 4. GUI FUNCTIONS ---
# (This section is unchanged)

def predict_digit():
    # 1. Process the in-memory PIL image
    processed_img = process_image(pil_image)

    if processed_img is None:
        prediction_label.config(text="Draw a digit!")
        return

    # 2. Convert to tensor and normalize
    img_tensor = transforms.ToTensor()(processed_img)
    mean = 0.1307
    std = 0.3081
    img_tensor = (img_tensor - mean) / std

    # 3. Add batch dimension and send to device
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 4. Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)

    # 5. Update the label
    prediction_label.config(text=f'Predicted: {predicted.item()}')


def clear_canvas():
    # Clear the on-screen canvas
    canvas.delete("all")

    # Also clear the in-memory PIL image
    pil_draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill="white")

    prediction_label.config(text="Draw a digit!")


def paint(event):
    pen_width = 15
    x1, y1 = (event.x - pen_width / 2), (event.y - pen_width / 2)
    x2, y2 = (event.x + pen_width / 2), (event.y + pen_width / 2)

    # Draw on the on-screen canvas
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")

    # Also draw on the in-memory PIL image
    pil_draw.ellipse([x1, y1, x2, y2], fill="black", outline="black")


# --- 5. SET UP AND RUN THE GUI ---
# (This section is unchanged)
if __name__ == "__main__":
    root = tk.Tk()
    root.title("MNIST Digit Recognizer")

    header_font = font.Font(family="Helvetica", size=18, weight="bold")
    button_font = font.Font(family="Helvetica", size=12)

    prediction_label = tk.Label(root, text="Draw a digit!", font=header_font, pady=10)
    prediction_label.pack()

    CANVAS_SIZE = 280

    # 1. Create the in-memory PIL image (must be RGB for ImageDraw)
    pil_image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
    # 2. Create the drawing object
    pil_draw = ImageDraw.Draw(pil_image)

    # Create the on-screen canvas
    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white", highlightthickness=1,
                       highlightbackground="black")
    canvas.pack()

    # Bind the paint function
    canvas.bind("<B1-Motion>", paint)

    button_frame = tk.Frame(root, pady=10)
    button_frame.pack()

    predict_button = tk.Button(button_frame, text="Predict", command=predict_digit, font=button_font)
    predict_button.pack(side=tk.LEFT, padx=10)

    clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas, font=button_font)
    clear_button.pack(side=tk.RIGHT, padx=10)

    # Start the application
    root.mainloop()