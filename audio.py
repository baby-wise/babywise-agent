import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Parámetros globales
# ----------------------------
SAMPLE_RATE = 22050
DURATION = 7
N_MFCC = 40
MAX_PAD_LEN = 130
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# CNNModel (igual que en entrenamiento)
# ----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.3)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.drop4 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop3(self.pool3(x))

        x = self.global_pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop4(x)
        return self.fc2(x)

# ----------------------------
# Extracción MFCC
# ----------------------------
def extract_mfcc(file_path, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.vstack([mfcc, delta, delta2])
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-10)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.astype(np.float32)
    except Exception as e:
        print(f"Error al procesar {file_path}: {e}")
        return None

# ----------------------------
# Cargar modelo
# ----------------------------
modelo = None
def load_llanto_model(path="modelopt.pth"):
    global modelo
    modelo = CNNModel().to(DEVICE)
    modelo.load_state_dict(torch.load(path, map_location=DEVICE))
    modelo.eval()  # Modo evaluación

# ----------------------------
# Predecir llanto
# ----------------------------
def predecir_llanto(file_path):
    global modelo
    if modelo is None:
        load_llanto_model()
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return "error"

    X = np.expand_dims(mfcc, axis=0)  # batch
    X = np.expand_dims(X, axis=0)     # channel
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        output = modelo(X_tensor)
        clase = torch.argmax(output, dim=1).item()

    return "cry" if clase == 1 else "not_cry"