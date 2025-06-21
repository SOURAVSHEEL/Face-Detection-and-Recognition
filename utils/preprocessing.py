def crop_face(image, bbox, margin=0.2):
    """
    Crop face region from image with optional margin.
    Args:
        image (np.ndarray): Original BGR image
        bbox (tuple): (x1, y1, x2, y2) bounding box
        margin (float): Expansion ratio
    Returns:
        face_crop (np.ndarray): Cropped face region
    """
    h, w, _ = image.shape
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    x1 = max(0, int(x1 - margin * width))
    y1 = max(0, int(y1 - margin * height))
    x2 = min(w, int(x2 + margin * width))
    y2 = min(h, int(y2 + margin * height))

    return image[y1:y2, x1:x2]
