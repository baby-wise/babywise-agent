import os
import logging

# Forzar a TensorFlow a no usar GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tempfile
import numpy as np
import librosa
from tensorflow.keras.models import load_model

logging.basicConfig(
    level=logging.DEBUG,  # muestra DEBUG, INFO, WARNING, ERROR
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("agent")

# Silenciar librerías ruidosas
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("librosa").setLevel(logging.WARNING)

# Parámetros MFCC y ventanas
SAMPLE_RATE = 22050
DURATION = 7
N_MFCC = 13
MAX_PAD_LEN = 130
AUDIO_WINDOW_SECONDS = 6
VIDEO_WINDOW_SECONDS = 2

modelo = None 
def load_llanto_model():
    global modelo
    modelo = load_model("modelo_llanto.keras")

def extract_mfcc(file_path, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        if len(audio) == 0:
            logger.error("El archivo de audio está vacío!")
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0), (0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc
    except Exception as e:
        logger.exception(f"Error al procesar {file_path}: {e}")
        return None

def predecir_llanto(file_path):
    global modelo
    if modelo is None:
        load_llanto_model()  # fallback por si no se prewarmeó
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return "error"
    X = mfcc[np.newaxis, ..., np.newaxis]
    y_pred = modelo.predict(X)
    clase = np.argmax(y_pred)
    return "cry" if clase == 1 else "not_cry"