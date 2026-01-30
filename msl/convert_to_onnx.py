"""
Convert PyTorch Model to ONNX
=============================
This converts your trained_model.pth to ONNX format
which can run directly in the browser with ONNX.js

Run this script:
    python convert_to_onnx.py

Output:
    msl_model.onnx (upload this to GitHub)
"""

import torch
import torch.nn as nn
import numpy as np

# Your model definition (same as training)
class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super(ImprovedLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.bn1(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Load model
print("Loading PyTorch model...")
model = ImprovedLSTM(258, 128, 30, dropout=0.3)
model.load_state_dict(torch.load('trained_model.pth', map_location='cpu'))
model.eval()
print("✅ Model loaded!")

# Create dummy input (batch=1, sequence=30, features=258)
dummy_input = torch.randn(1, 30, 258)

# Export to ONNX
print("Converting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    "msl_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✅ Saved: msl_model.onnx")
print("\nUpload msl_model.onnx to your GitHub repo!")
