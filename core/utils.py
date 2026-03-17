import base64
import cv2
import numpy as np

def decode_image(data: str) -> np.ndarray:
    """Decodifica base64 para OpenCV"""
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def encode_image(img: np.ndarray) -> str:
    """Codifica imagem Processada de volta para base64 do formato JPG"""
    _, buffer = cv2.imencode('.jpg', img)
    out_b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{out_b64}"
