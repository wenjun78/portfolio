"""
MSL Recognition - Streamlit App
================================
Deploy for FREE on Streamlit Cloud:
1. Push to GitHub repo
2. Go to share.streamlit.io
3. Connect your repo
4. Done! Always online.

Files needed in repo:
- app.py (this file)
- trained_model.pth
- requirements.txt
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import mediapipe as mp
from collections import deque

# Page config
st.set_page_config(page_title="MSL Recognition", page_icon="🤟", layout="wide")

# Model Definition
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

# Labels
gestures = ['abang','perlahan','berapa','keluarga','apa','kakak','bagaimana','hujan','siapa','saudara','lemak','main','pukul','marah','buat','hari','pinjam','hi','bomba','ribut','panas','jahat','ayah','masalah','beli','emak','apa_khabar','kereta','mana','sejuk']
translations = {'abang':'Elder Brother','perlahan':'Slow','berapa':'How Much','keluarga':'Family','apa':'What','kakak':'Elder Sister','bagaimana':'How','hujan':'Rain','siapa':'Who','saudara':'Relative','lemak':'Fat','main':'Play','pukul':'Hit','marah':'Angry','buat':'Do/Make','hari':'Day','pinjam':'Borrow','hi':'Hi','bomba':'Fire Brigade','ribut':'Storm','panas':'Hot','jahat':'Evil','ayah':'Father','masalah':'Problem','beli':'Buy','emak':'Mother','apa_khabar':'How Are You','kereta':'Car','mana':'Where','sejuk':'Cold'}

# Load model
@st.cache_resource
def load_model():
    model = ImprovedLSTM(258, 128, 30, dropout=0.3)
    model.load_state_dict(torch.load('trained_model.pth', map_location='cpu', weights_only=False))
    model.eval()
    return model

model = load_model()

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.keypoints_buffer = deque(maxlen=30)
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.prediction = None
        self.top3 = []
    
    def extract_keypoints(self, results):
        pose = np.zeros(33 * 4)
        lh = np.zeros(21 * 3)
        rh = np.zeros(21 * 3)
        
        if results.pose_landmarks:
            pose = np.array([[l.x, l.y, l.z, l.visibility] for l in results.pose_landmarks.landmark]).flatten()
        if results.left_hand_landmarks:
            lh = np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks:
            rh = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten()
        
        return np.concatenate([pose, lh, rh])
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)
        
        # Draw landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,170,255), thickness=2),
                mp_drawing.DrawingSpec(color=(0,170,255), thickness=2))
        
        # Extract and buffer keypoints
        kp = self.extract_keypoints(results)
        self.keypoints_buffer.append(kp)
        
        # Predict if buffer full and hands detected
        has_hands = results.left_hand_landmarks or results.right_hand_landmarks
        if has_hands and len(self.keypoints_buffer) >= 30:
            sequence = np.array(list(self.keypoints_buffer), dtype=np.float32)
            
            with torch.no_grad():
                input_tensor = torch.tensor(sequence[np.newaxis, ...])
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
                
                self.top3 = []
                for i in range(3):
                    idx = top3_indices[0][i].item()
                    prob = top3_probs[0][i].item() * 100
                    self.top3.append({
                        'gesture': gestures[idx],
                        'english': translations[gestures[idx]],
                        'confidence': round(prob, 1)
                    })
                
                self.prediction = self.top3[0]
        
        # Draw prediction on frame
        if self.prediction:
            cv2.putText(img, f"{self.prediction['gesture'].upper()}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(img, f"{self.prediction['english']} ({self.prediction['confidence']}%)", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw buffer status
        buffer_pct = len(self.keypoints_buffer) / 30
        cv2.rectangle(img, (20, img.shape[0]-30), (220, img.shape[0]-10), (50,50,50), -1)
        cv2.rectangle(img, (20, img.shape[0]-30), (int(20 + 200*buffer_pct), img.shape[0]-10), (138,43,226), -1)
        cv2.putText(img, f"{len(self.keypoints_buffer)}/30", (230, img.shape[0]-15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI
st.title("🤟 Malaysian Sign Language Recognition")
st.markdown("**Real-time Detection • 30 Gestures • 98% Accuracy**")

col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="msl",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("🎯 Top 3 Predictions")
    
    if ctx.video_processor:
        # Show predictions
        if ctx.video_processor.top3:
            for i, pred in enumerate(ctx.video_processor.top3):
                medal = ["🥇", "🥈", "🥉"][i]
                st.metric(
                    label=f"{medal} {pred['gesture'].replace('_', ' ')}",
                    value=f"{pred['confidence']}%",
                    delta=pred['english']
                )
    
    st.subheader("📖 Instructions")
    st.markdown("""
    1. Allow camera access
    2. Show your hands
    3. Perform a gesture
    4. Hold for 1-2 seconds
    """)
    
    st.subheader("🎨 Hand Colors")
    st.markdown("🟢 Green = Left Hand")
    st.markdown("🔵 Blue = Right Hand")

st.markdown("---")
st.markdown("**WQF7006 Computer Vision** — Group 13, Universiti Malaya")
