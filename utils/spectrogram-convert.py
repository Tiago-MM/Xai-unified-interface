from datasets import load_dataset
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def save_spec(audio_data, sr, save_path):
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100) # Taille optimisée pour VGG16
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    ms = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(save_path)
    plt.close(fig)




ds = load_dataset("Hemg/Deepfake-Audio-Dataset")
base_path = "./spectrograms"
os.makedirs(os.path.join(base_path, "REAL"), exist_ok=True)
os.makedirs(os.path.join(base_path, "FAKE"), exist_ok=True)


for i, item in enumerate(ds['train']):
    label_dir = "REAL" if item['label'] == 0 else "FAKE"
    
    file_path = os.path.join(base_path, label_dir, f"audio_{i}.png")
    
    # Extraction des données audio du dataset
    audio_array = item['audio']['array']
    sampling_rate = item['audio']['sampling_rate']
    
    save_spec(audio_array, sampling_rate, file_path)