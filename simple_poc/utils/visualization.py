import base64

import cv2


def img_to_base64(img):
    """Convert an OpenCV image to base64 encoded string for HTML embedding"""
    _, buffer = cv2.imencode(".jpg", img)
    img_str = buffer.tobytes()
    return base64.b64encode(img_str).decode("utf-8")
