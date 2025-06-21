import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the face detection model
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_faces(image):
    """
    Detect faces in a given image using MediaPipe.

    Args:
        image (np.ndarray): Input image in BGR format.

    Returns:
        list: List of bounding boxes in (x1, y1, x2, y2) format.
    """
    if image is None or image.size == 0:
        return []

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(image_rgb)

    bboxes = []
    if results.detections:
        h, w, _ = image.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int((bboxC.xmin + bboxC.width) * w)
            y2 = int((bboxC.ymin + bboxC.height) * h)

            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bboxes.append((x1, y1, x2, y2))

    return bboxes
