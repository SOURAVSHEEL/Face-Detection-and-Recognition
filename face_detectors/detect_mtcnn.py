import cv2
from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(image):
    """
    Detect faces in a single image using MTCNN.

    Args:
        image (np.ndarray): Input image (BGR format from OpenCV)

    Returns:
        list: List of bounding boxes (x1, y1, x2, y2)
    """
    if image is None or image.size == 0:
        return []

    # Convert BGR (OpenCV) to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        detections = detector.detect_faces(image_rgb)
    except Exception as e:
        print(f"[WARN] MTCNN failed: {e}")
        return []

    results = []
    for det in detections:
        if det['confidence'] < 0.90:  # optional threshold
            continue
        x, y, width, height = det['box']
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = x1 + width
        y2 = y1 + height
        results.append((x1, y1, x2, y2))

    return results
