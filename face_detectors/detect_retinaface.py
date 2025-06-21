import cv2
from insightface.app import FaceAnalysis


app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def detect_faces(image):
    """
    Detect faces in the given image using InsightFace.
    Args:
        image (numpy.ndarray): The input image in BGR format.
    Returns:
        list: List of detected faces with bounding boxes in (x1, y1, x2, y2) format.
    """
    results = []
    faces = app.get(image)
    
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        results.append((x1, y1, x2, y2))
    
    return results

