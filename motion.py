import numpy as np
import cv2
from livekit.rtc._proto import video_frame_pb2 as proto_video

def frame_to_rgb_array(frame):
    rgb_frame = frame.convert(proto_video.VideoBufferType.RGB24)
    arr = np.frombuffer(rgb_frame.data, dtype=np.uint8)
    arr = arr.reshape((rgb_frame.height, rgb_frame.width, 3))
    return arr

def detectar_movimiento(prev_frame, current_frame, umbral=2_000_000):
    """
    Detecta movimiento entre dos frames consecutivos.
    prev_frame, current_frame: objetos de tipo VideoFrame (LiveKit)
    umbral: valor absoluto de diferencia para considerar que hay movimiento
    Retorna True si hay movimiento, False si no.
    """
    arr1 = frame_to_rgb_array(prev_frame)
    arr2 = frame_to_rgb_array(current_frame)
    gray1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(arr2, cv2.COLOR_RGB2GRAY)
    shape = (256, 256)
    gray1 = cv2.resize(gray1, shape)
    gray2 = cv2.resize(gray2, shape)
    diff = cv2.absdiff(gray1, gray2)
    suma = np.sum(diff)
    print(f"[detectar_movimiento] suma diferencias: {suma}")
    return suma > umbral