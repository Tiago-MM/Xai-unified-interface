import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2

# --- TRAITEMENT AUDIO (Repo 1) ---
def process_audio(uploaded_file):
    # Charger l'audio .wav
    y, sr = librosa.load(io.BytesIO(uploaded_file.read()))
    # Conversion en spectrogramme pour détection Deepfake
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Préparation pour le CNN
    img = cv2.resize(S_db, (224, 224))
    img_rgb = np.stack([img]*3, axis=-1)
    img_final = np.expand_dims(img_rgb, axis=0)
    return img_final, S_db