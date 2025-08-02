import streamlit as st
import cv2
import numpy as np
import faiss
import pickle
import torch
import os
import tempfile
from datetime import datetime
import logging
from pathlib import Path
import time

# Import your modules
from face_detectors import detect_mediapipe as detector
from feature_extractors.facenet_extractor import FaceNetEmbedder
from anti_spoofing.scr.deepPixBiS_model import DeepPiXBiS
from preprocess import preprocess_image

# Page configuration
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and alignment
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Status boxes with better alignment */
    .status-box {
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 2px solid #bee5eb;
        color: #0c5460;
        text-align: left;
    }
    
    /* Button container alignment */
    .button-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    }
    
    /* Custom button styling */
    .stButton > button {
        width: 100%;
        min-height: 3rem;
        font-weight: 600;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1f77b4, #2e8bc0);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    
    /* Secondary button styling */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(90deg, #6c757d, #5a6268);
        color: white;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        text-align: center;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 2px solid #dee2e6;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 3px solid #1f77b4;
    }
    
    /* Camera frame container */
    .camera-container {
        border: 3px solid #1f77b4;
        border-radius: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* User list styling */
    .user-list {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    
    /* Center content */
    .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    
    /* Registration stats */
    .registration-stats {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Instructions panel */
    .instructions-panel {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1.5rem 0;
        border-left: 5px solid #1f77b4;
        color: black;  
    }
            
    .instructions-panel h4 {
        color: black;
    }
            
    .instructions-panel ol {
        color: black;
    }
    
    /* Detection info styling */
    .detection-info {
        background-color: #fff3cd;
        border: 2px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .button-container {
            flex-direction: column;
            align-items: center;
        }
        
        .stButton > button {
            min-width: 200px;
        }
    }
</style>
""", unsafe_allow_html=True)

class FaceRecognitionApp:
    def __init__(self):
        self.setup_session_state()
        
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'page' not in st.session_state:
            st.session_state.page = 'inference'
        if 'registered_users' not in st.session_state:
            st.session_state.registered_users = self.load_registered_users()
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'registration_count' not in st.session_state:
            st.session_state.registration_count = 0
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
            
    def load_registered_users(self):
        """Load list of registered users"""
        try:
            data_dir = Path("data")
            if data_dir.exists():
                return [folder.name for folder in data_dir.iterdir() if folder.is_dir()]
            return []
        except Exception as e:
            st.error(f"Error loading registered users: {e}")
            return []
    
    @st.cache_resource
    def load_models(_self):
        """Load anti-spoofing and recognition models"""
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load anti-spoofing model
            spoof_model = DeepPiXBiS().to(device)
            model_path = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\model\deepPixBiS_final.pth"
            spoof_model.load_state_dict(torch.load(model_path, map_location=device))
            spoof_model.eval()
            
            # Load FAISS index and embedder for recognition
            embedder = None
            faiss_index = None
            label_map = None
            
            try:
                faiss_index = faiss.read_index("faiss_index/index.bin")
                with open("faiss_index/labels.pkl", "rb") as f:
                    label_map = pickle.load(f)
                embedder = FaceNetEmbedder()
            except Exception as e:
                st.warning("Recognition model not found. Only registration will be available.")
            
            return {
                'spoof_model': spoof_model,
                'device': device,
                'embedder': embedder,
                'faiss_index': faiss_index,
                'label_map': label_map
            }
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None
    
    def is_real_face(self, face_crop, spoof_model, device, threshold=0.4):
        """Check if face is real using anti-spoofing model"""
        try:
            face_tensor = preprocess_image(face_crop).to(device)
            with torch.no_grad():
                _, global_pred = spoof_model(face_tensor)
                raw = torch.sigmoid(global_pred).item()
                return raw > threshold, raw
        except Exception as e:
            st.error(f"Error in face verification: {e}")
            return False, 0.0
    
    def recognize_face(self, face_crop, models, threshold=1.0):
        """Recognize face using FAISS index"""
        try:
            if not models['embedder'] or not models['faiss_index']:
                return "Model not loaded", 999.0
                
            emb = models['embedder'].get_embedding(face_crop).astype("float32")
            emb = np.expand_dims(emb, axis=0)
            D, I = models['faiss_index'].search(emb, k=1)
            dist, idx = D[0][0], I[0][0]
            
            if dist < threshold:
                return models['label_map'][idx], dist
            else:
                return "Unknown", dist
        except Exception as e:
            st.error(f"Error in face recognition: {e}")
            return "Error", 999.0
    
    def create_save_directory(self, name):
        """Create directory for saving registered faces"""
        save_dir = os.path.join("data", name)
        try:
            os.makedirs(save_dir, exist_ok=True)
            return save_dir
        except Exception as e:
            st.error(f"Failed to create directory {save_dir}: {e}")
            return None
    
    def save_face_image(self, face_crop, save_dir, name, count):
        """Save face image with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{name}_{count:04d}_{timestamp}.jpg"
            save_path = os.path.join(save_dir, filename)
            
            success = cv2.imwrite(save_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                return True, save_path
            return False, ""
        except Exception as e:
            st.error(f"Error saving face image: {e}")
            return False, ""
    
    def registration_page(self):
        """Face registration interface with improved alignment"""
        st.markdown('<div class="main-header">👤 Face Registration</div>', unsafe_allow_html=True)
        
        # Check if name came from sidebar
        sidebar_name = getattr(st.session_state, 'sidebar_registration_name', '')
        
        # Centered input section
        st.markdown('<div class="center-content">', unsafe_allow_html=True)
        
        # User input with better spacing
        col2, col3 = st.columns([2, 1])
        with col2:
            name = st.text_input(
                "", 
                value=sidebar_name,  # Pre-fill with sidebar name
                placeholder="👤 Enter full name for registration", 
                label_visibility="collapsed"
            )
        
        with col3:
            if st.button("🏠 Back to Inference", type="secondary"):
                st.session_state.page = 'inference'
                st.session_state.camera_active = False
                # Clear sidebar name
                if hasattr(st.session_state, 'sidebar_registration_name'):
                    del st.session_state.sidebar_registration_name
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not name or not name.strip():
            st.markdown("""
            <div class="info-box instructions-panel">
            <h4>📝 Registration Instructions</h4>
            <p>Please enter a valid name above to begin the registration process.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Clean name validation
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
        if not clean_name:
            st.markdown('<div class="error-box">❌ Name contains invalid characters</div>', unsafe_allow_html=True)
            return
        
        # Check if user exists
        if clean_name in st.session_state.registered_users:
            st.markdown(f'<div class="error-box">⚠️ User "{clean_name}" already exists!</div>', unsafe_allow_html=True)
            return
        
        # Registration header
        st.markdown(f"""
        <div class="success-box">
        <h3>Registering: {clean_name}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera controls with better alignment
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            start_camera = st.button("📹 Start Camera", type="primary", use_container_width=True)
        with col2:
            stop_camera = st.button("⏹️ Stop Camera", type="secondary", use_container_width=True)
        with col3:
            save_frame_clicked = st.button("💾 Save Frame", type="secondary", 
                                        disabled=not st.session_state.camera_active, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle button clicks
        if start_camera:
            st.session_state.camera_active = True
            st.session_state.registration_count = 0
        if stop_camera:
            st.session_state.camera_active = False
        
        # Instructions
        st.markdown("""
        <div class="instructions-panel">
        <h4>📋 Registration Steps:</h4>
        <ol>
            <li>Click <strong>"Start Camera"</strong> to begin</li>
            <li>Position your face clearly in the camera view</li>
            <li>Wait for <span style="color: green; font-weight: bold;">GREEN box</span> (Real face detected)</li>
            <li>Click <strong>"Save Frame"</strong> to capture the image</li>
            <li>Repeat 10-15 times for better accuracy</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera feed container
        if st.session_state.camera_active:
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            st.markdown("### 📷 Live Camera Feed")
            self.run_registration_camera(clean_name, save_frame_clicked)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Registration progress
        if st.session_state.registration_count > 0:
            st.markdown(f"""
            <div class="registration-stats">
            📊 Registration Progress: {st.session_state.registration_count} images saved
            </div>
            """, unsafe_allow_html=True)
    
    def run_registration_camera(self, name, save_frame_clicked):
        """Run camera for registration with continuous live preview"""
        models = self.load_models()
        if not models:
            st.error("Models not loaded!")
            return
        
        save_dir = self.create_save_directory(name)
        if not save_dir:
            st.error("Could not create save directory!")
            return
        
        # Camera setup
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            # Continuous loop for live video
            frame_count = 0
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Process every few frames to reduce computation
                if frame_count % 3 == 0:  # Process every 3rd frame
                    # Detect faces
                    boxes = detector.detect_faces(frame)
                    display_frame = frame.copy()
                    
                    face_status = "No faces detected"
                    real_face_detected = False
                    current_face_crop = None
                    
                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        # Expand bounding box
                        margin = 0.15
                        width, height = x2 - x1, y2 - y1
                        x1_exp = max(int(x1 - margin * width), 0)
                        y1_exp = max(int(y1 - margin * height), 0)
                        x2_exp = min(int(x2 + margin * width), frame.shape[1])
                        y2_exp = min(int(y2 + margin * height), frame.shape[0])
                        
                        face_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                        
                        if face_crop.size > 0:
                            # Check if face is real
                            real, score = self.is_real_face(face_crop, models['spoof_model'], models['device'])
                            
                            color = (0, 255, 0) if real else (0, 0, 255)
                            label = f"Real ({score:.3f})" if real else f"Fake ({score:.3f})"
                            face_status = f"Face {i+1}: {label}"
                            
                            if real:
                                real_face_detected = True
                                current_face_crop = face_crop
                            
                            cv2.rectangle(display_frame, (x1_exp, y1_exp), (x2_exp, y2_exp), color, 2)
                            cv2.putText(display_frame, label, (x1_exp, y1_exp - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Save frame if button was clicked and real face is detected
                    if save_frame_clicked and real_face_detected and current_face_crop is not None:
                        success, save_path = self.save_face_image(current_face_crop, save_dir, name, st.session_state.registration_count)
                        if success:
                            st.session_state.registration_count += 1
                            if name not in st.session_state.registered_users:
                                st.session_state.registered_users.append(name)
                            status_placeholder.success(f"✅ Saved image {st.session_state.registration_count}")
                    
                    # Add text overlay
                    cv2.putText(display_frame, f"Saved: {st.session_state.registration_count} | {face_status}",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Convert BGR to RGB for Streamlit
                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
                
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            st.error(f"Error in registration camera: {e}")
        finally:
            cap.release()
    
    def inference_page(self):
        """Face recognition inference interface with improved alignment"""
        st.markdown('<div class="main-header">🔍 Face Recognition System</div>', unsafe_allow_html=True)
        
        # Top navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # with col1:
        #     if st.button("➕ Register New User", type="primary", use_container_width=True):
        #         st.session_state.page = 'registration'
        #         st.session_state.camera_active = False
        #         st.rerun()
        
        with col2:
            st.markdown('<div class="center-content"><h3>Live Recognition Mode</h3></div>', unsafe_allow_html=True)
        
        # Camera controls
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            start_recognition = st.button("📹 Start Recognition", type="primary", use_container_width=True)
        with col2:
            stop_camera = st.button("⏹️ Stop Camera", type="secondary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle button clicks
        if start_recognition:
            st.session_state.camera_active = True
        if stop_camera:
            st.session_state.camera_active = False
        
        # Camera feed
        if st.session_state.camera_active:
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            st.markdown("### 📷 Live Recognition Feed")
            self.run_inference_camera()
            st.markdown('</div>', unsafe_allow_html=True)

    
    def run_inference_camera(self):
        """Run camera for inference with continuous live recognition"""
        models = self.load_models()
        if not models:
            st.error("Models not loaded!")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        try:
            # Continuous loop for live video
            frame_count = 0
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Process every few frames to reduce computation
                if frame_count % 3 == 0:  # Process every 3rd frame
                    # Detect faces
                    boxes = detector.detect_faces(frame)
                    display_frame = frame.copy()
                    
                    detection_info = []
                    
                    for i, (x1, y1, x2, y2) in enumerate(boxes):
                        # Expand bounding box
                        margin = 0.2
                        width, height = x2 - x1, y2 - y1
                        x1_exp = max(int(x1 - margin * width), 0)
                        y1_exp = max(int(y1 - margin * height), 0)
                        x2_exp = min(int(x2 + margin * width), frame.shape[1])
                        y2_exp = min(int(y2 + margin * height), frame.shape[0])
                        
                        face_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                        
                        if face_crop.size > 0:
                            # Check if face is real
                            real, spoof_score = self.is_real_face(face_crop, models['spoof_model'], models['device'])
                            
                            if real:
                                # Recognize face
                                name, dist = self.recognize_face(face_crop, models)
                                label = f"{name} ({dist:.2f})" if name != "Unknown" else "Unknown"
                                color = (0, 255, 0) if name != "Unknown" else (255, 0, 0)
                                detection_info.append(f"Face {i+1}: {label}")
                            else:
                                label = "Spoof Detected"
                                color = (0, 0, 255)
                                detection_info.append(f"Face {i+1}: Spoofing attempt")
                            
                            cv2.rectangle(display_frame, (x1_exp, y1_exp), (x2_exp, y2_exp), color, 2)
                            cv2.putText(display_frame, label, (x1_exp, y1_exp - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Add detection info to frame
                    info_text = f"Faces: {len(boxes)} | " + " | ".join(detection_info[:2])  # Limit text length
                    cv2.putText(display_frame, info_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Convert BGR to RGB for Streamlit
                    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Show detection stats
                    if detection_info:
                        stats_text = "**Current Detections:**\n" + "\n".join([f"• {info}" for info in detection_info])
                        stats_placeholder.markdown(stats_text)
                    else:
                        stats_placeholder.markdown("**👀 Looking for faces...**")
                
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            st.error(f"Error in inference camera: {e}")
        finally:
            cap.release()
    
    def run(self):
        """Main application runner"""
        # Enhanced Sidebar
        with st.sidebar:
            st.markdown("## 🎛️ System Controls")
            st.markdown(f"**Current Mode:** {st.session_state.page.title()}")
            
            if st.button("🔄 Refresh Users"):
                st.session_state.registered_users = self.load_registered_users()
                st.success("User list refreshed!")
            
            # Registration section in sidebar
            st.markdown("---")
            st.markdown("## ➕ New User Registration")

            # Registration form in sidebar
            with st.container():
                sidebar_name = st.text_input(
                    "👤 Enter name:", 
                    placeholder="Full name", 
                    key="sidebar_name_input"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📝 Register", type="primary", use_container_width=True):
                        if sidebar_name and sidebar_name.strip():
                            clean_name = "".join(c for c in sidebar_name if c.isalnum() or c in (' ', '_', '-')).strip()
                            if clean_name and clean_name not in st.session_state.registered_users:
                                st.session_state.page = 'registration'
                                st.session_state.camera_active = False
                                # Store the name for the registration page
                                st.session_state.sidebar_registration_name = clean_name
                                st.rerun()
                            elif clean_name in st.session_state.registered_users:
                                st.error(f"User '{clean_name}' exists!")
                            else:
                                st.error("Invalid name!")
                        else:
                            st.error("Enter a name!")
                
                with col2:
                    if st.button("🔍 Inference", type="secondary", use_container_width=True):
                        st.session_state.page = 'inference'
                        st.session_state.camera_active = False
                        st.rerun()
            
            st.markdown("---")
            st.markdown("## 📊 System Info")
            st.markdown(f"**👥 Registered Users:** {len(st.session_state.registered_users)}")
            st.markdown(f"**📷 Camera:** {'🟢 Active' if st.session_state.camera_active else '🔴 Inactive'}")
            
            # # Show registered users list
            # if st.session_state.registered_users:
            #     st.markdown("### 👥 Users:")
            #     for i, user in enumerate(st.session_state.registered_users, 1):
            #         st.markdown(f"**{i}.** {user}")
            
            if st.session_state.page == 'registration':
                st.markdown("---")
                st.markdown("## 📸 Registration Stats")
                st.markdown(f"**Images Saved:** {st.session_state.registration_count}")
                
                # Add quick save button in sidebar during registration
                if st.session_state.camera_active:
                    if st.button("💾 Quick Save", type="primary", use_container_width=True):
                        st.session_state.save_frame = True
        
        # Main content
        if st.session_state.page == 'inference':
            self.inference_page()
        else:
            self.registration_page()


# Main execution
def main():
    app = FaceRecognitionApp()
    app.run()

if __name__ == "__main__":
    main()
