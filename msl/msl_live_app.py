"""
🤟 MSL Live Detection - Local Flask App
========================================
USAGE:
    1. Put this file in same folder as trained_model.pth
    2. Run: python msl_live_app.py
    3. Open: http://localhost:5000
    4. Click "Start Camera" and show your hands!

REQUIREMENTS:
    pip install flask flask-cors torch numpy
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import os

# ============================================
# MODEL DEFINITION
# ============================================
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

# ============================================
# GESTURE LABELS
# ============================================
GESTURES = [
    'abang', 'perlahan', 'berapa', 'keluarga', 'apa',
    'kakak', 'bagaimana', 'hujan', 'siapa', 'saudara',
    'lemak', 'main', 'pukul', 'marah', 'buat',
    'hari', 'pinjam', 'hi', 'bomba', 'ribut',
    'panas', 'jahat', 'ayah', 'masalah', 'beli',
    'emak', 'apa_khabar', 'kereta', 'mana', 'sejuk'
]

TRANSLATIONS = {
    'abang': 'Elder Brother', 'perlahan': 'Slow', 'berapa': 'How Much',
    'keluarga': 'Family', 'apa': 'What', 'kakak': 'Elder Sister',
    'bagaimana': 'How', 'hujan': 'Rain', 'siapa': 'Who', 'saudara': 'Relative',
    'lemak': 'Fat', 'main': 'Play', 'pukul': 'Hit', 'marah': 'Angry',
    'buat': 'Do/Make', 'hari': 'Day', 'pinjam': 'Borrow', 'hi': 'Hi',
    'bomba': 'Fire Brigade', 'ribut': 'Storm', 'panas': 'Hot', 'jahat': 'Evil',
    'ayah': 'Father', 'masalah': 'Problem', 'beli': 'Buy', 'emak': 'Mother',
    'apa_khabar': 'How Are You', 'kereta': 'Car', 'mana': 'Where', 'sejuk': 'Cold'
}

# ============================================
# LOAD MODEL
# ============================================
print("=" * 50)
print("🤟 MSL Live Detection")
print("=" * 50)

model = None
model_loaded = False

# Try different possible model paths
model_paths = [
    'trained_model.pth',
    'msl/trained_model.pth',
    '../trained_model.pth',
]

for path in model_paths:
    if os.path.exists(path):
        print(f"📁 Found model: {path}")
        try:
            model = ImprovedLSTM(258, 128, 30, dropout=0.3)
            try:
                model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
            except TypeError:
                model.load_state_dict(torch.load(path, map_location='cpu'))
            model.eval()
            model_loaded = True
            print("✅ Model loaded successfully!")
            break
        except Exception as e:
            print(f"❌ Error loading model: {e}")

if not model_loaded:
    print("⚠️  Model not found! Please put 'trained_model.pth' in the same folder.")
    print("    The app will run in DEMO MODE with simulated predictions.")

print("=" * 50)

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)
CORS(app)

# ============================================
# HTML TEMPLATE - Beautiful UI
# ============================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤟 MSL Live Detection</title>
    
    <!-- MediaPipe -->
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2d1b4e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Header */
        .header {
            text-align: center;
            margin-bottom: 25px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        
        .header p {
            color: rgba(255,255,255,0.7);
            font-size: 1.1rem;
        }
        
        .badges {
            margin-top: 15px;
            display: flex;
            justify-content: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .badge {
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .badge-green { background: linear-gradient(135deg, #22c55e, #16a34a); }
        .badge-blue { background: linear-gradient(135deg, #3b82f6, #2563eb); }
        .badge-purple { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }
        .badge-orange { background: linear-gradient(135deg, #f59e0b, #d97706); }
        
        /* Model Status */
        .model-status {
            text-align: center;
            padding: 12px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        .model-status.ready {
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid #22c55e;
            color: #4ade80;
        }
        
        .model-status.demo {
            background: rgba(251, 191, 36, 0.2);
            border: 1px solid #fbbf24;
            color: #fcd34d;
        }
        
        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
        }
        
        @media (max-width: 900px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        
        /* Video Section */
        .video-section {
            background: linear-gradient(145deg, #1e1e3f, #2a2a5a);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 15px 50px rgba(0,0,0,0.4);
        }
        
        .video-container {
            position: relative;
            width: 100%;
            border-radius: 16px;
            overflow: hidden;
            background: #000;
            aspect-ratio: 4/3;
        }
        
        video { display: none; }
        
        canvas {
            width: 100%;
            height: 100%;
            display: block;
        }
        
        /* Loading Overlay */
        .loading-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.85);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 10;
        }
        
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(255,255,255,0.2);
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* Status Bar */
        .status-bar {
            position: absolute;
            top: 0; left: 0; right: 0;
            padding: 15px 20px;
            background: linear-gradient(180deg, rgba(0,0,0,0.85), transparent);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 5;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #ef4444;
            transition: all 0.3s;
        }
        
        .status-dot.active {
            background: #22c55e;
            box-shadow: 0 0 15px #22c55e;
        }
        
        .buffer-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .buffer-bar {
            width: 120px;
            height: 10px;
            background: rgba(255,255,255,0.15);
            border-radius: 5px;
            overflow: hidden;
        }
        
        .buffer-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.1s;
            border-radius: 5px;
        }
        
        /* Result Overlay */
        .result-overlay {
            position: absolute;
            bottom: 0; left: 0; right: 0;
            padding: 25px;
            background: linear-gradient(0deg, rgba(0,0,0,0.95) 60%, transparent);
            text-align: center;
            z-index: 5;
        }
        
        .result-gesture {
            font-size: 2.8rem;
            font-weight: 800;
            color: #4ade80;
            text-shadow: 0 0 30px rgba(74, 222, 128, 0.5);
            margin-bottom: 5px;
        }
        
        .result-english {
            font-size: 1.2rem;
            color: rgba(255,255,255,0.9);
            margin-bottom: 15px;
        }
        
        /* Top 3 Cards */
        .top3-cards {
            display: flex;
            justify-content: center;
            gap: 12px;
            flex-wrap: wrap;
        }
        
        .top3-card {
            background: rgba(255,255,255,0.08);
            padding: 12px 18px;
            border-radius: 12px;
            text-align: center;
            min-width: 110px;
            border: 2px solid transparent;
            transition: all 0.3s;
        }
        
        .top3-card.rank1 {
            background: rgba(34, 197, 94, 0.25);
            border-color: #22c55e;
        }
        
        .top3-card.rank2 {
            background: rgba(59, 130, 246, 0.2);
            border-color: #3b82f6;
        }
        
        .top3-card.rank3 {
            background: rgba(168, 85, 247, 0.2);
            border-color: #a855f7;
        }
        
        .top3-card .gesture-name {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 3px;
        }
        
        .top3-card .gesture-english {
            font-size: 0.75rem;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .top3-card .gesture-conf {
            font-size: 1.2rem;
            font-weight: 800;
        }
        
        .top3-card.rank1 .gesture-conf { color: #4ade80; }
        .top3-card.rank2 .gesture-conf { color: #60a5fa; }
        .top3-card.rank3 .gesture-conf { color: #c084fc; }
        
        /* Controls */
        .controls {
            display: flex;
            gap: 12px;
            margin-top: 20px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        button {
            padding: 16px 35px;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-start {
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: white;
        }
        
        .btn-reset {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
        }
        
        /* Sidebar */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        
        .card {
            background: linear-gradient(145deg, #1e1e3f, #2a2a5a);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .card h3 {
            font-size: 1.1rem;
            margin-bottom: 15px;
            color: #a5b4fc;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .card ul {
            list-style: none;
        }
        
        .card li {
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            font-size: 0.95rem;
        }
        
        .card li:last-child { border-bottom: none; }
        
        /* Sidebar Top 3 */
        .sidebar-top3 .top3-row {
            display: flex;
            justify-content: space-between;
            padding: 12px 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            margin-bottom: 8px;
            align-items: center;
        }
        
        .sidebar-top3 .top3-row:last-child { margin-bottom: 0; }
        
        /* Gestures Grid */
        .gestures-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .gesture-item {
            background: rgba(255,255,255,0.05);
            padding: 8px 10px;
            border-radius: 8px;
            font-size: 0.8rem;
        }
        
        .gesture-item strong {
            color: #c4b5fd;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: rgba(255,255,255,0.5);
            font-size: 0.9rem;
        }
        
        .footer a {
            color: #a5b4fc;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🤟 Malaysian Sign Language Recognition</h1>
            <p>Real-time Gesture Detection with BiLSTM Neural Network</p>
            <div class="badges">
                <span class="badge badge-green">✅ 98% Accuracy</span>
                <span class="badge badge-blue">📷 Live Camera</span>
                <span class="badge badge-purple">🧠 30 Gestures</span>
                <span class="badge badge-orange">⚡ Real-time</span>
            </div>
        </div>
        
        <!-- Model Status -->
        <div class="model-status {{ 'ready' if model_loaded else 'demo' }}" id="modelStatus">
            {% if model_loaded %}
                ✅ Model loaded successfully! Ready for real predictions.
            {% else %}
                ⚠️ Demo mode - Model not found. Place 'trained_model.pth' in the same folder and restart.
            {% endif %}
        </div>
        
        <!-- Main Grid -->
        <div class="main-grid">
            <!-- Video Section -->
            <div class="video-section">
                <div class="video-container">
                    <video id="webcam" playsinline></video>
                    <canvas id="canvas"></canvas>
                    
                    <!-- Loading -->
                    <div class="loading-overlay" id="loading">
                        <div class="spinner"></div>
                        <p style="margin-top: 20px; font-size: 1.1rem;">Loading MediaPipe...</p>
                    </div>
                    
                    <!-- Status Bar -->
                    <div class="status-bar" id="statusBar" style="display: none;">
                        <div class="status-indicator">
                            <span class="status-dot" id="statusDot"></span>
                            <span id="statusText">Waiting...</span>
                        </div>
                        <div class="buffer-container">
                            <span id="bufferText">0/30</span>
                            <div class="buffer-bar">
                                <div class="buffer-fill" id="bufferFill" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Results -->
                    <div class="result-overlay" id="resultOverlay" style="display: none;">
                        <div class="result-gesture" id="resultGesture">---</div>
                        <div class="result-english" id="resultEnglish">Show your hands to start</div>
                        <div class="top3-cards" id="top3Cards">
                            <div class="top3-card rank1">
                                <div class="gesture-name">---</div>
                                <div class="gesture-english">---</div>
                                <div class="gesture-conf">--%</div>
                            </div>
                            <div class="top3-card rank2">
                                <div class="gesture-name">---</div>
                                <div class="gesture-english">---</div>
                                <div class="gesture-conf">--%</div>
                            </div>
                            <div class="top3-card rank3">
                                <div class="gesture-name">---</div>
                                <div class="gesture-english">---</div>
                                <div class="gesture-conf">--%</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Controls -->
                <div class="controls">
                    <button class="btn-start" id="startBtn" onclick="startCamera()">
                        📷 Start Camera
                    </button>
                    <button class="btn-reset" onclick="resetApp()">
                        🔄 Reset
                    </button>
                </div>
            </div>
            
            <!-- Sidebar -->
            <div class="sidebar">
                <!-- Instructions -->
                <div class="card">
                    <h3>📖 How to Use</h3>
                    <ul>
                        <li>1. Click <strong>Start Camera</strong></li>
                        <li>2. Allow camera access</li>
                        <li>3. Show your hands clearly</li>
                        <li>4. Perform a gesture</li>
                        <li>5. Hold for 1-2 seconds</li>
                    </ul>
                </div>
                
                <!-- Top 3 Predictions -->
                <div class="card sidebar-top3">
                    <h3>🎯 Top 3 Predictions</h3>
                    <div id="sidebarTop3">
                        <div class="top3-row"><span>---</span><span>--%</span></div>
                        <div class="top3-row"><span>---</span><span>--%</span></div>
                        <div class="top3-row"><span>---</span><span>--%</span></div>
                    </div>
                </div>
                
                <!-- Hand Colors -->
                <div class="card">
                    <h3>🎨 Hand Colors</h3>
                    <ul>
                        <li>🟢 <strong>Green</strong> = Left Hand</li>
                        <li>🔵 <strong>Blue</strong> = Right Hand</li>
                    </ul>
                </div>
                
                <!-- Gestures List -->
                <div class="card">
                    <h3>📚 30 Gestures</h3>
                    <div class="gestures-grid" id="gesturesGrid"></div>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p><strong>WQF7006 Computer Vision</strong> — Group 13, Universiti Malaya</p>
            <p style="margin-top: 5px;">MediaPipe Holistic • BiLSTM Model • PyTorch</p>
        </div>
    </div>

    <script>
        // ============================================
        // CONFIGURATION
        // ============================================
        const GESTURES = {{ gestures | tojson }};
        const TRANSLATIONS = {{ translations | tojson }};
        
        // Populate gestures grid
        document.getElementById('gesturesGrid').innerHTML = GESTURES.map(g => 
            `<div class="gesture-item"><strong>${g.replace('_', ' ')}</strong><br><small>${TRANSLATIONS[g]}</small></div>`
        ).join('');
        
        // ============================================
        // VARIABLES
        // ============================================
        let holistic;
        let running = false;
        let keypoints = [];
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // ============================================
        // MEDIAPIPE INITIALIZATION
        // ============================================
        async function initMediaPipe() {
            holistic = new Holistic({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
            });
            
            holistic.setOptions({
                modelComplexity: 0,
                smoothLandmarks: true,
                minDetectionConfidence: 0.5,
                minTrackingConfidence: 0.5
            });
            
            holistic.onResults(onResults);
            
            document.getElementById('loading').querySelector('p').textContent = 'Ready! Click Start Camera';
        }
        
        // ============================================
        // EXTRACT KEYPOINTS
        // ============================================
        function extractKeypoints(results) {
            let pose = new Array(132).fill(0);
            let leftHand = new Array(63).fill(0);
            let rightHand = new Array(63).fill(0);
            
            if (results.poseLandmarks) {
                pose = results.poseLandmarks.flatMap(lm => [lm.x, lm.y, lm.z, lm.visibility || 0]);
            }
            if (results.leftHandLandmarks) {
                leftHand = results.leftHandLandmarks.flatMap(lm => [lm.x, lm.y, lm.z]);
            }
            if (results.rightHandLandmarks) {
                rightHand = results.rightHandLandmarks.flatMap(lm => [lm.x, lm.y, lm.z]);
            }
            
            return [...pose, ...leftHand, ...rightHand];
        }
        
        // ============================================
        // PROCESS RESULTS
        // ============================================
        function onResults(results) {
            if (!running) return;
            
            // Draw video frame
            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
            
            // Draw hand landmarks
            if (results.leftHandLandmarks) {
                drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 3});
                drawLandmarks(ctx, results.leftHandLandmarks, {color: '#00FF00', radius: 5});
            }
            if (results.rightHandLandmarks) {
                drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, {color: '#00AAFF', lineWidth: 3});
                drawLandmarks(ctx, results.rightHandLandmarks, {color: '#00AAFF', radius: 5});
            }
            
            ctx.restore();
            
            // Update status
            const hasHands = results.leftHandLandmarks || results.rightHandLandmarks;
            document.getElementById('statusDot').className = hasHands ? 'status-dot active' : 'status-dot';
            document.getElementById('statusText').textContent = hasHands ? '✓ Hands Detected' : 'Show your hands...';
            
            // Add keypoints to buffer
            keypoints.push(extractKeypoints(results));
            if (keypoints.length > 30) keypoints.shift();
            
            // Update buffer display
            const bufferPct = (keypoints.length / 30) * 100;
            document.getElementById('bufferFill').style.width = bufferPct + '%';
            document.getElementById('bufferText').textContent = keypoints.length + '/30';
            
            // Predict if we have enough frames and hands are detected
            if (hasHands && keypoints.length >= 30) {
                predict();
            }
        }
        
        // ============================================
        // PREDICTION
        // ============================================
        let lastPrediction = 0;
        
        async function predict() {
            // Throttle predictions to every 300ms
            if (Date.now() - lastPrediction < 300) return;
            lastPrediction = Date.now();
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ keypoints: keypoints })
                });
                
                const data = await response.json();
                
                if (data.top3) {
                    updateUI(data.top3);
                }
            } catch (error) {
                console.error('Prediction error:', error);
            }
        }
        
        // ============================================
        // UPDATE UI
        // ============================================
        function updateUI(top3) {
            // Main result
            const best = top3[0];
            document.getElementById('resultGesture').textContent = best.gesture.toUpperCase().replace('_', ' ');
            document.getElementById('resultEnglish').textContent = `${best.english} (${best.confidence}%)`;
            
            // Top 3 cards on video
            document.getElementById('top3Cards').innerHTML = top3.map((item, i) => `
                <div class="top3-card rank${i + 1}">
                    <div class="gesture-name">${item.gesture.replace('_', ' ')}</div>
                    <div class="gesture-english">${item.english}</div>
                    <div class="gesture-conf">${item.confidence}%</div>
                </div>
            `).join('');
            
            // Sidebar top 3
            document.getElementById('sidebarTop3').innerHTML = top3.map(item => `
                <div class="top3-row">
                    <span>${item.gesture.replace('_', ' ')}</span>
                    <span><strong>${item.confidence}%</strong></span>
                </div>
            `).join('');
        }
        
        // ============================================
        // START CAMERA
        // ============================================
        async function startCamera() {
            const btn = document.getElementById('startBtn');
            btn.disabled = true;
            btn.innerHTML = '⏳ Starting...';
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });
                
                video.srcObject = stream;
                await video.play();
                
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                
                running = true;
                
                // Show UI elements
                document.getElementById('loading').style.display = 'none';
                document.getElementById('statusBar').style.display = 'flex';
                document.getElementById('resultOverlay').style.display = 'block';
                
                btn.innerHTML = '✅ Running';
                
                // Start processing loop
                async function processFrame() {
                    if (!running) return;
                    await holistic.send({ image: video });
                    requestAnimationFrame(processFrame);
                }
                processFrame();
                
            } catch (error) {
                alert('Camera error: ' + error.message);
                btn.disabled = false;
                btn.innerHTML = '📷 Start Camera';
            }
        }
        
        // ============================================
        // RESET
        // ============================================
        function resetApp() {
            keypoints = [];
            
            document.getElementById('bufferFill').style.width = '0%';
            document.getElementById('bufferText').textContent = '0/30';
            document.getElementById('resultGesture').textContent = '---';
            document.getElementById('resultEnglish').textContent = 'Show your hands to start';
            
            document.getElementById('top3Cards').innerHTML = `
                <div class="top3-card rank1">
                    <div class="gesture-name">---</div>
                    <div class="gesture-english">---</div>
                    <div class="gesture-conf">--%</div>
                </div>
                <div class="top3-card rank2">
                    <div class="gesture-name">---</div>
                    <div class="gesture-english">---</div>
                    <div class="gesture-conf">--%</div>
                </div>
                <div class="top3-card rank3">
                    <div class="gesture-name">---</div>
                    <div class="gesture-english">---</div>
                    <div class="gesture-conf">--%</div>
                </div>
            `;
            
            document.getElementById('sidebarTop3').innerHTML = `
                <div class="top3-row"><span>---</span><span>--%</span></div>
                <div class="top3-row"><span>---</span><span>--%</span></div>
                <div class="top3-row"><span>---</span><span>--%</span></div>
            `;
        }
        
        // ============================================
        // INITIALIZE
        // ============================================
        initMediaPipe();
    </script>
</body>
</html>
'''

# ============================================
# ROUTES
# ============================================
@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        model_loaded=model_loaded,
        gestures=GESTURES,
        translations=TRANSLATIONS
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    kp = data.get('keypoints', [])
    
    if len(kp) < 30:
        return jsonify({'status': 'collecting', 'message': 'Need more frames'})
    
    # Use real model if loaded, otherwise demo mode
    if model_loaded and model is not None:
        sequence = np.array(kp[-30:], dtype=np.float32)
        
        with torch.no_grad():
            input_tensor = torch.tensor(sequence[np.newaxis, ...], dtype=torch.float32)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
            
            top3 = []
            for i in range(3):
                idx = top3_indices[0][i].item()
                prob = top3_probs[0][i].item() * 100
                top3.append({
                    'gesture': GESTURES[idx],
                    'english': TRANSLATIONS[GESTURES[idx]],
                    'confidence': round(prob, 1)
                })
            
            return jsonify({'status': 'success', 'top3': top3})
    else:
        # Demo mode - return simulated predictions
        import random
        last = kp[-1]
        lh = last[132:195]
        rh = last[195:258]
        has_left = any(v != 0 for v in lh)
        has_right = any(v != 0 for v in rh)
        
        if has_right and sum(lh[i] for i in range(1, len(lh), 3)) / 21 < 0.4:
            choices = [('hi', 'Hi'), ('apa_khabar', 'How Are You'), ('panas', 'Hot')]
        elif has_left and has_right:
            choices = [('keluarga', 'Family'), ('saudara', 'Relative'), ('apa_khabar', 'How Are You')]
        elif has_right:
            choices = [('apa', 'What'), ('siapa', 'Who'), ('mana', 'Where')]
        else:
            choices = [('abang', 'Elder Brother'), ('kakak', 'Elder Sister'), ('ayah', 'Father')]
        
        top3 = []
        conf = 85
        for g, e in choices:
            top3.append({
                'gesture': g,
                'english': e,
                'confidence': conf + random.randint(-5, 5)
            })
            conf -= 25
        
        return jsonify({'status': 'demo', 'top3': top3})

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("\n🚀 Starting server...")
    print("📍 Open in browser: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
