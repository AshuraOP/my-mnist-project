import torch
import torch.nn as nn

# --- 1. DEFINE THE MODEL ARCHITECTURE ---
# This is the corrected version.
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
        # The typo was here (--1), this is now fixed (-1)
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. LOAD YOUR TRAINED WEIGHTS ---
PYTORCH_MODEL_PATH = 'mnist_cnn_model.pth'
ONNX_MODEL_PATH = 'mnist_model.onnx' # The new file we will create

device = torch.device('cpu') # No need for CUDA here
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- 3. EXPORT TO ONNX ---
print(f"Loading model from {PYTORCH_MODEL_PATH}...")

# Create a dummy input tensor that matches the model's input
# (1 sample, 1 color channel, 28x28 pixels)
dummy_input = torch.randn(1, 1, 28, 28, device='cpu')

print("Exporting model to ONNX format...")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_MODEL_PATH,
    input_names=['input'],   # We name the input 'input'
    output_names=['output'], # We name the output 'output'
    opset_version=11
)

print(f"Model successfully converted and saved to {ONNX_MODEL_PATH}")