import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def preprocess_audio(audio_file):
    # 1. Charger l'audio
    y, sr = librosa.load(audio_file, duration=3.0) 
    
    # 2. Créer le spectrogramme (Mel-spectrogram)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 3. Normalisation/Resize pour le modèle (ex: 224x224 pour VGG16)
    # Cette étape dépend de l'input_shape de votre modèle chargé
    return S_db