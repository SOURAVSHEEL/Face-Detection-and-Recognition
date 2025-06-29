import cv2
import os
import argparse
from datetime import datetime
import logging

# Face detection using MediaPipe
from face_detectors import detect_mediapipe as detector

# Anti-spoofing model
import torch
from anti_spoofing.scr.deepPixBiS_model import DeepPiXBiS
from preprocess import preprocess_image

# Setup logger
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler with formatting
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

logger = setup_logger()

# Load anti-spoofing model
def load_antispoofing_model(model_path=None):
    """Load the anti-spoofing model with proper error handling"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        spoof_model = DeepPiXBiS().to(device)
        
        # Default model path if not provided
        if model_path is None:
            model_path = r"C:\Users\soura\OneDrive\Desktop\Projects\Face-Detection-and-Recognition\anti_spoofing\model\deepPixBiS_v3.pth"
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, None
            
        spoof_model.load_state_dict(torch.load(model_path, map_location=device))
        spoof_model.eval()
        logger.info(f"Anti-spoofing model loaded successfully from: {model_path}")
        
        return spoof_model, device
    except Exception as e:
        logger.error(f"Failed to load anti-spoofing model: {e}")
        return None, None

def is_real_face(face_crop, spoof_model, device, threshold=0.4):
    """Check if face is real or fake using anti-spoofing model"""
    try:
        face_tensor = preprocess_image(face_crop).to(device)
        with torch.no_grad():
            _, global_pred = spoof_model(face_tensor)
            raw = torch.sigmoid(global_pred).item()
            return raw > threshold, raw
    except Exception as e:
        logger.error(f"Error in face verification: {e}")
        return False, 0.0

def create_save_directory(name):
    """Create directory for saving faces with proper error handling"""
    save_dir = os.path.join("data", name)
    try:
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Save directory created/confirmed: {save_dir}")
        
        # Test write permissions
        test_file = os.path.join(save_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.info("Directory write permissions confirmed")
        
        return save_dir
    except Exception as e:
        logger.error(f"Failed to create or access directory {save_dir}: {e}")
        return None

def save_face_image(face_crop, save_dir, name, count):
    """Save face image with proper error handling"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"{name}_{count:04d}_{timestamp}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        # Ensure face crop is valid
        if face_crop is None or face_crop.size == 0:
            logger.warning("Invalid face crop - cannot save")
            return False, ""
            
        # Save with high quality
        success = cv2.imwrite(save_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if success:
            # Verify file was actually created and has size > 0
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                logger.info(f"Successfully saved: {save_path}")
                logger.info(f"File size: {os.path.getsize(save_path)} bytes")
                return True, save_path
            else:
                logger.error(f"File was not created properly: {save_path}")
                return False, ""
        else:
            logger.error(f"cv2.imwrite failed for: {save_path}")
            return False, ""
            
    except Exception as e:
        logger.error(f"Error saving face image: {e}")
        return False, ""

def register(name, model_path=None):
    """Main registration function"""
    logger.info(f"Starting registration for: {name}")
    
    # Load anti-spoofing model
    spoof_model, device = load_antispoofing_model(model_path)
    if spoof_model is None:
        logger.error("Cannot proceed without anti-spoofing model")
        return
    
    # Create save directory
    save_dir = create_save_directory(name)
    if save_dir is None:
        logger.error("Cannot proceed without valid save directory")
        return

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
        
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    count = 0
    logger.info("=== FACE REGISTRATION STARTED ===")
    logger.info("Instructions:")
    logger.info("- Position your face in the camera view")
    logger.info("- Wait for GREEN box (Real face detected)")
    logger.info("- Press 'S' to save the face")
    logger.info("- Press 'Q' to quit")
    logger.info("=====================================")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read from camera")
                break

            # Detect faces
            boxes = detector.detect_faces(frame)
            
            display_frame = frame.copy()
            
            # Process each detected face
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                try:
                    # Expand bounding box with margin
                    margin = 0.15  # Slightly larger margin
                    width, height = x2 - x1, y2 - y1
                    x1_exp = max(int(x1 - margin * width), 0)
                    y1_exp = max(int(y1 - margin * height), 0)
                    x2_exp = min(int(x2 + margin * width), frame.shape[1])
                    y2_exp = min(int(y2 + margin * height), frame.shape[0])
                    
                    # Extract face crop
                    face_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
                    
                    if face_crop.size == 0:
                        logger.warning(f"Empty face crop for box {i}")
                        continue

                    # Check if face is real
                    real, score = is_real_face(face_crop, spoof_model, device)

                    # Draw bounding box and label
                    color = (0, 255, 0) if real else (0, 0, 255)  # Green for real, red for fake
                    label = f"Real ({score:.3f})" if real else f"Fake ({score:.3f})"

                    cv2.rectangle(display_frame, (x1_exp, y1_exp), (x2_exp, y2_exp), color, 2)
                    cv2.putText(display_frame, label, (x1_exp, y1_exp - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Add face number if multiple faces
                    if len(boxes) > 1:
                        cv2.putText(display_frame, f"Face {i+1}", (x1_exp, y2_exp + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                except Exception as e:
                    logger.error(f"Error processing face {i}: {e}")
                    continue

            # Add instruction text on frame
            cv2.putText(display_frame, f"Saved: {count} faces | Press 'S' to save | 'Q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow(f"Registration: {name}", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') or key == ord('S'):
                logger.info("=== SAVE KEY PRESSED ===")
                logger.info(f"Detected faces: {len(boxes)}")
                
                if not boxes:
                    logger.warning("No faces detected - cannot save")
                    continue
                
                saved_count = 0
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    try:
                        # Use same processing as display
                        margin = 0.15
                        width, height = x2 - x1, y2 - y1
                        x1_exp = max(int(x1 - margin * width), 0)
                        y1_exp = max(int(y1 - margin * height), 0)
                        x2_exp = min(int(x2 + margin * width), frame.shape[1])
                        y2_exp = min(int(y2 + margin * height), frame.shape[0])
                        
                        face_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]

                        if face_crop.size == 0:
                            logger.warning(f"Face {i+1}: Empty crop - skipping")
                            continue

                        # Verify face is real
                        real, score = is_real_face(face_crop, spoof_model, device)
                        logger.info(f"Face {i+1}: Real={real}, Score={score:.3f}, Shape={face_crop.shape}")
                        
                        if real:
                            success, save_path = save_face_image(face_crop, save_dir, name, count)
                            if success:
                                count += 1
                                saved_count += 1
                                logger.info(f"✓ Face {i+1} saved successfully (Total: {count})")
                            else:
                                logger.error(f"✗ Failed to save face {i+1}")
                        else:
                            logger.warning(f"Face {i+1}: Fake face detected (score: {score:.3f}) - not saved")
                            
                    except Exception as e:
                        logger.error(f"Error saving face {i+1}: {e}")
                        continue

                if saved_count == 0:
                    logger.warning("No real faces were saved this time")
                else:
                    logger.info(f"Successfully saved {saved_count} face(s)")
                    
                logger.info("========================")

            elif key == ord('q') or key == ord('Q'):
                logger.info("Quit key pressed - exiting registration")
                break
                
    except KeyboardInterrupt:
        logger.info("Registration interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Registration completed. Total faces saved: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Registration Tool with Anti-Spoofing")
    parser.add_argument("--name", required=True, help="Name of the person to register")
    parser.add_argument("--model", help="Path to anti-spoofing model file")
    args = parser.parse_args()
    
    # Validate name parameter
    if not args.name or not args.name.strip():
        print("Error: Name cannot be empty")
        exit(1)
        
    # Clean name (remove special characters)
    clean_name = "".join(c for c in args.name if c.isalnum() or c in (' ', '_', '-')).strip()
    if not clean_name:
        print("Error: Name contains only invalid characters")
        exit(1)
        
    register(clean_name, args.model)