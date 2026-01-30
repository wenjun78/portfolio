"""
🤟 MSL Live Detection - Local Flask App v2
===========================================
FEATURES:
    - Live camera detection (predictions in TOP RIGHT corner)
    - Video upload processing
    
USAGE:
    1. Put this file in same folder as trained_model.pth
    2. Run: python msl_live_app_v2.py
    3. Open: http://localhost:5000

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
print("🤟 MSL Live Detection v2")
print("=" * 50)

model = None
model_loaded = False

model_paths = ['trained_model.pth', 'msl/trained_model.pth', '../trained_model.pth']

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
    print("⚠️  Model not found! Running in DEMO MODE.")

print("=" * 50)

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)
CORS(app)

# ============================================
# HTML TEMPLATE
# ============================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤟 MSL Live Detection</title>
    
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic/holistic.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js" crossorigin="anonymous"></script>
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #2d1b4e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 15px;
        }
        
        .container { max-width: 1200px; margin: 0 auto; }
        
        /* Header */
        .header { text-align: center; margin-bottom: 20px; }
        .header h1 {
            font-size: 2.2rem;
            background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .header p { color: rgba(255,255,255,0.7); }
        .badges { margin-top: 10px; display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
        .badge { padding: 5px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }
        .badge-green { background: #22c55e; }
        .badge-blue { background: #3b82f6; }
        .badge-purple { background: #8b5cf6; }
        
        /* Model Status */
        .model-status { text-align: center; padding: 10px; border-radius: 8px; margin-bottom: 15px; font-size: 0.9rem; }
        .model-status.ready { background: rgba(34,197,94,0.2); border: 1px solid #22c55e; }
        .model-status.demo { background: rgba(251,191,36,0.2); border: 1px solid #fbbf24; }
        
        /* Tabs */
        .tabs { display: flex; justify-content: center; gap: 10px; margin-bottom: 20px; }
        .tab-btn {
            padding: 12px 30px; font-size: 1rem; font-weight: 600;
            border: none; border-radius: 10px; cursor: pointer;
            background: rgba(255,255,255,0.1); color: #fff; transition: 0.3s;
        }
        .tab-btn:hover { background: rgba(255,255,255,0.2); }
        .tab-btn.active { background: linear-gradient(135deg, #667eea, #764ba2); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        /* Main Grid */
        .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }
        
        /* Video Section */
        .video-section { background: linear-gradient(145deg, #1e1e3f, #2a2a5a); border-radius: 16px; padding: 15px; }
        .video-container { position: relative; width: 100%; border-radius: 12px; overflow: hidden; background: #000; aspect-ratio: 4/3; }
        video { display: none; }
        canvas { width: 100%; height: 100%; display: block; }
        
        /* Loading */
        .loading-overlay {
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.85); display: flex;
            flex-direction: column; align-items: center; justify-content: center;
        }
        .spinner { width: 50px; height: 50px; border: 4px solid rgba(255,255,255,0.2); border-top-color: #667eea; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        /* Status Bar - TOP LEFT */
        .status-bar {
            position: absolute; top: 10px; left: 10px;
            padding: 8px 12px; background: rgba(0,0,0,0.7);
            border-radius: 8px; display: flex; align-items: center; gap: 10px;
        }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; background: #ef4444; }
        .status-dot.active { background: #22c55e; box-shadow: 0 0 10px #22c55e; }
        
        /* Buffer Bar - BOTTOM LEFT */
        .buffer-box {
            position: absolute; bottom: 10px; left: 10px;
            padding: 8px 12px; background: rgba(0,0,0,0.7);
            border-radius: 8px; display: flex; align-items: center; gap: 10px;
        }
        .buffer-bar { width: 100px; height: 8px; background: rgba(255,255,255,0.2); border-radius: 4px; overflow: hidden; }
        .buffer-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.1s; }
        
        /* Result Box - TOP RIGHT (doesn't block view) */
        .result-box {
            position: absolute; top: 10px; right: 10px;
            background: rgba(0,0,0,0.85); border-radius: 10px;
            padding: 10px 15px; min-width: 150px;
            border: 2px solid #22c55e;
        }
        .result-gesture { font-size: 1.3rem; font-weight: 800; color: #4ade80; text-align: center; }
        .result-english { font-size: 0.8rem; color: rgba(255,255,255,0.8); text-align: center; }
        .result-conf { font-size: 1rem; font-weight: 700; color: #fbbf24; text-align: center; margin-top: 3px; }
        
        /* Controls */
        .controls { display: flex; gap: 10px; margin-top: 15px; justify-content: center; flex-wrap: wrap; }
        button { padding: 14px 28px; font-size: 1rem; font-weight: 600; border: none; border-radius: 10px; cursor: pointer; transition: 0.3s; }
        button:hover { transform: translateY(-2px); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-start { background: linear-gradient(135deg, #22c55e, #16a34a); color: #fff; }
        .btn-reset { background: linear-gradient(135deg, #6366f1, #4f46e5); color: #fff; }
        .btn-upload { background: linear-gradient(135deg, #f59e0b, #d97706); color: #fff; }
        
        /* Sidebar */
        .sidebar { display: flex; flex-direction: column; gap: 15px; }
        .card { background: linear-gradient(145deg, #1e1e3f, #2a2a5a); border-radius: 12px; padding: 15px; }
        .card h3 { font-size: 1rem; margin-bottom: 12px; color: #a5b4fc; }
        .card ul { list-style: none; }
        .card li { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 0.9rem; }
        .card li:last-child { border-bottom: none; }
        
        /* Top 3 rows */
        .top3-row { display: flex; justify-content: space-between; padding: 10px 12px; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 6px; }
        .top3-row.rank1 { background: rgba(34,197,94,0.2); border-left: 3px solid #22c55e; }
        .top3-row.rank2 { background: rgba(59,130,246,0.15); border-left: 3px solid #3b82f6; }
        .top3-row.rank3 { background: rgba(168,85,247,0.15); border-left: 3px solid #a855f7; }
        
        /* Upload Area */
        .upload-area { border: 2px dashed rgba(255,255,255,0.3); border-radius: 12px; padding: 50px; text-align: center; cursor: pointer; transition: 0.3s; }
        .upload-area:hover { border-color: #667eea; background: rgba(102,126,234,0.1); }
        
        /* Gestures Grid */
        .gestures-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; max-height: 180px; overflow-y: auto; }
        .gesture-item { background: rgba(255,255,255,0.05); padding: 6px 8px; border-radius: 6px; font-size: 0.75rem; }
        .gesture-item strong { color: #c4b5fd; }
        
        .footer { text-align: center; margin-top: 25px; color: rgba(255,255,255,0.5); font-size: 0.85rem; }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🤟 Malaysian Sign Language Recognition</h1>
        <p>Real-time Gesture Detection • BiLSTM Neural Network</p>
        <div class="badges">
            <span class="badge badge-green">✅ 98% Accuracy</span>
            <span class="badge badge-blue">📷 Live Camera</span>
            <span class="badge badge-purple">🧠 30 Gestures</span>
        </div>
    </div>
    
    <div class="model-status {{ 'ready' if model_loaded else 'demo' }}">
        {% if model_loaded %}✅ Model loaded! Ready for real predictions.{% else %}⚠️ Demo mode - Place 'trained_model.pth' in folder and restart.{% endif %}
    </div>
    
    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('live')">📷 Live Detection</button>
        <button class="tab-btn" onclick="switchTab('upload')">📁 Video Upload</button>
    </div>
    
    <!-- LIVE DETECTION TAB -->
    <div class="tab-content active" id="tab-live">
        <div class="main-grid">
            <div class="video-section">
                <div class="video-container">
                    <video id="webcam" playsinline></video>
                    <canvas id="canvas"></canvas>
                    
                    <div class="loading-overlay" id="loading">
                        <div class="spinner"></div>
                        <p style="margin-top:15px">Loading MediaPipe...</p>
                    </div>
                    
                    <!-- Status - TOP LEFT -->
                    <div class="status-bar" id="statusBar" style="display:none">
                        <span class="status-dot" id="statusDot"></span>
                        <span id="statusText">Waiting...</span>
                    </div>
                    
                    <!-- Buffer - BOTTOM LEFT -->
                    <div class="buffer-box" id="bufferBox" style="display:none">
                        <span id="bufferText">0/30</span>
                        <div class="buffer-bar"><div class="buffer-fill" id="bufferFill"></div></div>
                    </div>
                    
                    <!-- Result - TOP RIGHT -->
                    <div class="result-box" id="resultBox" style="display:none">
                        <div class="result-gesture" id="resultGesture">---</div>
                        <div class="result-english" id="resultEnglish">---</div>
                        <div class="result-conf" id="resultConf">--%</div>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="btn-start" id="startBtn" onclick="startCamera()">📷 Start Camera</button>
                    <button class="btn-reset" onclick="resetLive()">🔄 Reset</button>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="card">
                    <h3>📖 How to Use</h3>
                    <ul>
                        <li>1. Click <strong>Start Camera</strong></li>
                        <li>2. Show your hands</li>
                        <li>3. Perform a gesture</li>
                        <li>4. Hold for 1-2 seconds</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>🎯 Top 3 Predictions</h3>
                    <div id="sidebarTop3">
                        <div class="top3-row rank1"><span>---</span><span>--%</span></div>
                        <div class="top3-row rank2"><span>---</span><span>--%</span></div>
                        <div class="top3-row rank3"><span>---</span><span>--%</span></div>
                    </div>
                </div>
                <div class="card">
                    <h3>🎨 Hand Colors</h3>
                    <ul><li>🟢 Green = Left Hand</li><li>🔵 Blue = Right Hand</li></ul>
                </div>
            </div>
        </div>
    </div>
    
    <!-- VIDEO UPLOAD TAB -->
    <div class="tab-content" id="tab-upload">
        <div class="main-grid">
            <div class="video-section">
                <div class="video-container">
                    <video id="uploadVideo" playsinline></video>
                    <canvas id="uploadCanvas"></canvas>
                    
                    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                        <div style="font-size:3rem">📁</div>
                        <p style="margin-top:15px"><strong>Click to upload video</strong></p>
                        <p style="font-size:0.85rem;opacity:0.7;margin-top:8px">MP4, WebM, MOV</p>
                    </div>
                    <input type="file" id="fileInput" accept="video/*" style="display:none">
                    
                    <div class="status-bar" id="uploadStatusBar" style="display:none">
                        <span class="status-dot active"></span>
                        <span id="uploadStatusText">Processing...</span>
                    </div>
                    
                    <div class="buffer-box" id="uploadBufferBox" style="display:none">
                        <span id="uploadProgress">0%</span>
                        <div class="buffer-bar"><div class="buffer-fill" id="uploadFill"></div></div>
                    </div>
                    
                    <div class="result-box" id="uploadResultBox" style="display:none">
                        <div class="result-gesture" id="uploadResultGesture">---</div>
                        <div class="result-english" id="uploadResultEnglish">---</div>
                        <div class="result-conf" id="uploadResultConf">--%</div>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="btn-upload" onclick="document.getElementById('fileInput').click()">📁 Choose Video</button>
                    <button class="btn-start" id="processBtn" onclick="processVideo()" disabled>🔍 Process</button>
                    <button class="btn-reset" onclick="resetUpload()">🔄 Reset</button>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="card">
                    <h3>📖 How to Use</h3>
                    <ul>
                        <li>1. Click <strong>Choose Video</strong></li>
                        <li>2. Select a video file</li>
                        <li>3. Click <strong>Process</strong></li>
                        <li>4. View results</li>
                    </ul>
                </div>
                <div class="card" id="uploadResultsCard" style="display:none">
                    <h3>📊 Results</h3>
                    <div id="uploadTop3">
                        <div class="top3-row rank1"><span>---</span><span>--%</span></div>
                        <div class="top3-row rank2"><span>---</span><span>--%</span></div>
                        <div class="top3-row rank3"><span>---</span><span>--%</span></div>
                    </div>
                    <p style="margin-top:10px;font-size:0.8rem;opacity:0.7" id="uploadStats"></p>
                </div>
                <div class="card">
                    <h3>📚 30 Gestures</h3>
                    <div class="gestures-grid" id="gesturesGrid"></div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer"><strong>WQF7006 Computer Vision</strong> — Group 13, Universiti Malaya</div>
</div>

<script>
const GESTURES = {{ gestures | tojson }};
const TRANSLATIONS = {{ translations | tojson }};
document.getElementById('gesturesGrid').innerHTML = GESTURES.map(g => `<div class="gesture-item"><strong>${g.replace('_', ' ')}</strong><br><small>${TRANSLATIONS[g]}</small></div>`).join('');

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`[onclick="switchTab('${tab}')"]`).classList.add('active');
    document.getElementById('tab-' + tab).classList.add('active');
}

// ========== LIVE DETECTION ==========
let holistic, running = false, keypoints = [];
const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

async function initMediaPipe() {
    holistic = new Holistic({locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`});
    holistic.setOptions({modelComplexity: 0, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5});
    holistic.onResults(onResults);
    document.getElementById('loading').querySelector('p').textContent = 'Ready!';
}

function extractKeypoints(r) {
    let p = Array(132).fill(0), l = Array(63).fill(0), rh = Array(63).fill(0);
    if (r.poseLandmarks) p = r.poseLandmarks.flatMap(x => [x.x, x.y, x.z, x.visibility || 0]);
    if (r.leftHandLandmarks) l = r.leftHandLandmarks.flatMap(x => [x.x, x.y, x.z]);
    if (r.rightHandLandmarks) rh = r.rightHandLandmarks.flatMap(x => [x.x, x.y, x.z]);
    return [...p, ...l, ...rh];
}

function onResults(r) {
    if (!running) return;
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(r.image, 0, 0, canvas.width, canvas.height);
    if (r.leftHandLandmarks) {
        drawConnectors(ctx, r.leftHandLandmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 3});
        drawLandmarks(ctx, r.leftHandLandmarks, {color: '#00FF00', radius: 4});
    }
    if (r.rightHandLandmarks) {
        drawConnectors(ctx, r.rightHandLandmarks, HAND_CONNECTIONS, {color: '#00AAFF', lineWidth: 3});
        drawLandmarks(ctx, r.rightHandLandmarks, {color: '#00AAFF', radius: 4});
    }
    ctx.restore();
    
    const hasHands = r.leftHandLandmarks || r.rightHandLandmarks;
    document.getElementById('statusDot').className = hasHands ? 'status-dot active' : 'status-dot';
    document.getElementById('statusText').textContent = hasHands ? '✓ Hands' : 'Show hands';
    
    keypoints.push(extractKeypoints(r));
    if (keypoints.length > 30) keypoints.shift();
    document.getElementById('bufferFill').style.width = (keypoints.length / 30 * 100) + '%';
    document.getElementById('bufferText').textContent = keypoints.length + '/30';
    
    if (hasHands && keypoints.length >= 30) predict();
}

let lastPred = 0;
async function predict() {
    if (Date.now() - lastPred < 300) return;
    lastPred = Date.now();
    try {
        const res = await fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({keypoints})});
        const data = await res.json();
        if (data.top3) {
            const best = data.top3[0];
            document.getElementById('resultGesture').textContent = best.gesture.toUpperCase().replace('_', ' ');
            document.getElementById('resultEnglish').textContent = best.english;
            document.getElementById('resultConf').textContent = best.confidence + '%';
            document.getElementById('sidebarTop3').innerHTML = data.top3.map((t, i) =>
                `<div class="top3-row rank${i+1}"><span>${t.gesture.replace('_', ' ')}</span><span><strong>${t.confidence}%</strong></span></div>`
            ).join('');
        }
    } catch (e) { console.error(e); }
}

async function startCamera() {
    const btn = document.getElementById('startBtn');
    btn.disabled = true; btn.textContent = '⏳ Starting...';
    try {
        const stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
        video.srcObject = stream;
        await video.play();
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        running = true;
        document.getElementById('loading').style.display = 'none';
        document.getElementById('statusBar').style.display = 'flex';
        document.getElementById('bufferBox').style.display = 'flex';
        document.getElementById('resultBox').style.display = 'block';
        btn.textContent = '✅ Running';
        async function loop() { if (!running) return; await holistic.send({image: video}); requestAnimationFrame(loop); }
        loop();
    } catch (e) { alert('Camera error: ' + e.message); btn.disabled = false; btn.textContent = '📷 Start Camera'; }
}

function resetLive() {
    keypoints = [];
    document.getElementById('bufferFill').style.width = '0%';
    document.getElementById('bufferText').textContent = '0/30';
    document.getElementById('resultGesture').textContent = '---';
    document.getElementById('resultEnglish').textContent = '---';
    document.getElementById('resultConf').textContent = '--%';
    document.getElementById('sidebarTop3').innerHTML = '<div class="top3-row rank1"><span>---</span><span>--%</span></div><div class="top3-row rank2"><span>---</span><span>--%</span></div><div class="top3-row rank3"><span>---</span><span>--%</span></div>';
}

// ========== VIDEO UPLOAD ==========
let uploadFile = null;
document.getElementById('fileInput').onchange = e => {
    if (e.target.files[0]) {
        uploadFile = e.target.files[0];
        document.getElementById('uploadArea').innerHTML = `<div style="font-size:3rem">🎬</div><p style="margin-top:10px"><strong>${uploadFile.name}</strong></p>`;
        document.getElementById('processBtn').disabled = false;
    }
};

async function processVideo() {
    if (!uploadFile) return;
    const uv = document.getElementById('uploadVideo'), uc = document.getElementById('uploadCanvas'), uctx = uc.getContext('2d');
    document.getElementById('uploadArea').style.display = 'none';
    uc.style.display = 'block';
    document.getElementById('uploadStatusBar').style.display = 'flex';
    document.getElementById('uploadBufferBox').style.display = 'flex';
    document.getElementById('uploadResultBox').style.display = 'block';
    document.getElementById('processBtn').disabled = true;
    
    uv.src = URL.createObjectURL(uploadFile);
    await new Promise(r => uv.onloadedmetadata = r);
    uc.width = uv.videoWidth || 640; uc.height = uv.videoHeight || 480;
    
    const duration = uv.duration, fps = 15, totalFrames = Math.floor(duration * fps);
    let uk = [], predictions = [], processed = 0;
    
    const vh = new Holistic({locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`});
    vh.setOptions({modelComplexity: 0, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5});
    
    vh.onResults(async r => {
        processed++;
        document.getElementById('uploadFill').style.width = (processed / totalFrames * 100) + '%';
        document.getElementById('uploadProgress').textContent = Math.round(processed / totalFrames * 100) + '%';
        document.getElementById('uploadStatusText').textContent = `Frame ${processed}/${totalFrames}`;
        uctx.drawImage(r.image, 0, 0, uc.width, uc.height);
        if (r.leftHandLandmarks) drawConnectors(uctx, r.leftHandLandmarks, HAND_CONNECTIONS, {color: '#0F0', lineWidth: 2});
        if (r.rightHandLandmarks) drawConnectors(uctx, r.rightHandLandmarks, HAND_CONNECTIONS, {color: '#0AF', lineWidth: 2});
        
        uk.push(extractKeypoints(r));
        if (uk.length > 30) uk.shift();
        if ((r.leftHandLandmarks || r.rightHandLandmarks) && uk.length >= 30) {
            try {
                const res = await fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({keypoints: uk})});
                const data = await res.json();
                if (data.top3) predictions.push(data.top3[0]);
            } catch (e) {}
        }
    });
    
    for (let i = 0; i < totalFrames; i++) {
        uv.currentTime = i / fps;
        await new Promise(r => setTimeout(r, 50));
        await vh.send({image: uv});
    }
    
    if (predictions.length > 0) {
        const counts = {};
        predictions.forEach(p => counts[p.gesture] = (counts[p.gesture] || 0) + 1);
        const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        const best = predictions.find(p => p.gesture === sorted[0][0]);
        document.getElementById('uploadResultGesture').textContent = best.gesture.toUpperCase().replace('_', ' ');
        document.getElementById('uploadResultEnglish').textContent = best.english;
        document.getElementById('uploadResultConf').textContent = best.confidence + '%';
        document.getElementById('uploadResultsCard').style.display = 'block';
        document.getElementById('uploadTop3').innerHTML = sorted.slice(0, 3).map(([g, c], i) =>
            `<div class="top3-row rank${i+1}"><span>${g.replace('_', ' ')}</span><span><strong>${Math.round(c / predictions.length * 100)}%</strong></span></div>`
        ).join('');
        document.getElementById('uploadStats').textContent = `${processed} frames, ${predictions.length} predictions`;
    } else {
        document.getElementById('uploadResultGesture').textContent = 'No Hands';
        document.getElementById('uploadResultEnglish').textContent = 'No hands detected';
    }
    document.getElementById('processBtn').disabled = false;
}

function resetUpload() {
    uploadFile = null;
    document.getElementById('uploadArea').style.display = 'block';
    document.getElementById('uploadArea').innerHTML = '<div style="font-size:3rem">📁</div><p style="margin-top:15px"><strong>Click to upload video</strong></p>';
    document.getElementById('uploadCanvas').style.display = 'none';
    document.getElementById('uploadStatusBar').style.display = 'none';
    document.getElementById('uploadBufferBox').style.display = 'none';
    document.getElementById('uploadResultBox').style.display = 'none';
    document.getElementById('uploadResultsCard').style.display = 'none';
    document.getElementById('processBtn').disabled = true;
}

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
    return render_template_string(HTML_TEMPLATE, model_loaded=model_loaded, gestures=GESTURES, translations=TRANSLATIONS)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    kp = data.get('keypoints', [])
    if len(kp) < 30: return jsonify({'status': 'collecting'})
    
    if model_loaded and model is not None:
        sequence = np.array(kp[-30:], dtype=np.float32)
        with torch.no_grad():
            input_tensor = torch.tensor(sequence[np.newaxis, ...], dtype=torch.float32)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
            top3 = [{'gesture': GESTURES[top3_indices[0][i].item()], 'english': TRANSLATIONS[GESTURES[top3_indices[0][i].item()]], 'confidence': round(top3_probs[0][i].item() * 100, 1)} for i in range(3)]
            return jsonify({'status': 'success', 'top3': top3})
    else:
        import random
        demo = [('hi', 'Hi'), ('apa', 'What'), ('keluarga', 'Family')]
        top3 = [{'gesture': g, 'english': e, 'confidence': 90 - i*30 + random.randint(-5,5)} for i, (g, e) in enumerate(demo)]
        return jsonify({'status': 'demo', 'top3': top3})

if __name__ == '__main__':
    print("\n🚀 Starting server...")
    print("📍 Open: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)
