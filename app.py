import streamlit as st
import os
import io
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from models.audio_models import load_unified_model
from explanations.xai_methods import get_gradcam, superimpose_heatmap, get_lime

# Configuration de l'environnement pour éviter les crashs sur Mac
os.environ["TF_USE_LEGACY_KERAS"] = "1"
@st.cache_resource


# --- TRAITEMENT IMAGE (CheXpert / Repo 2) ---
def process_chest_xray(uploaded_file):
    # Lecture de l'image médicale
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    # Redimensionnement standard pour CNN
    img_resized = cv2.resize(img_array, (224, 224))
    img_final = np.expand_dims(img_resized, axis=0)
    # Normalisation pour modèles Keras
    img_final = img_final / 255.0
    return img_final, image

# --- TRAITEMENT AUDIO (Repo 1) ---
def process_audio(uploaded_file):
    # Charger l'audio .wav
    y, sr = librosa.load(io.BytesIO(uploaded_file.read()), duration=3.0)
    # Conversion en spectrogramme pour détection Deepfake
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Préparation pour le CNN
    img = cv2.resize(S_db, (224, 224))
    img_rgb = np.stack([img]*3, axis=-1)
    img_final = np.expand_dims(img_rgb, axis=0)
    return img_final, S_db

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="XAI Unified Interface", layout="wide")
st.title("🛡️ Unified Explainable AI Interface")

st.sidebar.header("1. Chargement des données")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier", type=['wav', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    
    # Vérification automatique de compatibilité
    if file_ext == 'wav':
        input_type = "Audio"
        available_models = ["VGG16 (Audio)", "ResNet", "MobileNet"]
        available_xai = ["Grad-CAM", "LIME", "SHAP"] # Requis pour audio
    else:
        input_type = "Image"
        available_models = ["AlexNet (Cancer)", "DenseNet"]
        available_xai = ["Grad-CAM", "LIME"] # Grad-CAM requis pour radiographies

    selected_model = st.sidebar.selectbox("Modèle de classification", available_models)
    selected_xai = st.sidebar.multiselect("Méthodes XAI à appliquer", available_xai)

    tab1, tab2 = st.tabs(["🚀 Analyse Unique", "📊 Comparaison XAI"])
    input_data = None
    model = None
    with tab1:
        if st.button("Lancer l'Analyse"):
            model = load_unified_model(selected_model)
            
            if input_type == "Audio":
                input_data, S_db = process_audio(uploaded_file)
                preds = model.predict(input_data)
                score = np.max(preds[0])
                label = "DEEPFAKE DETECTED" if score > 0.5 else "AUTHENTIC AUDIO"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Verdict Audio", label, f"Confiance: {score*100:.2f}%")
                    fig, ax = plt.subplots()
                    librosa.display.specshow(S_db, ax=ax)
                    st.pyplot(fig)
                with col2:
                    st.subheader("Explication")
                    st.info("Visualisation des fréquences suspectes via spectrogramme.")

            else: # Traitement IMAGE (Repo 2)
                input_data, original_img = process_chest_xray(uploaded_file)
                preds = model.predict(input_data)
                score = np.max(preds[0])
                label = "MALIGNANT" if score > 0.5 else "BENIGN / NORMAL"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Verdict Médical", label, f"Probabilité: {score*100:.2f}%")
                    st.image(original_img, caption="Radiographie originale", use_container_width=True)
                with col2:
                    st.subheader("Explication Grad-CAM")
                    st.warning("Identification des zones tumorales suspectes dans les poumons.")


    with tab2:
        st.subheader("Analyse comparative Side-by-Side")
        if uploaded_file and len(selected_xai) >= 1:
            conv_layer = "block5_conv3" if "VGG16" in selected_model else "conv5_block3_out"
            cols = st.columns(len(selected_xai))
            
            for i, method in enumerate(selected_xai):
                with cols[i]:
                    st.info(f"Méthode : {method}")
                    
                    if method == "Grad-CAM":
                        heatmap = get_gradcam(model, input_data, conv_layer)
                        base_img = cv2.resize(np.stack([S_db]*3, axis=-1) if input_type=="Audio" else np.array(original_img), (224,224))
                        result = superimpose_heatmap(heatmap, base_img)
                        st.image(result, caption="Heatmap Grad-CAM", use_container_width=True)
                    
                    elif method == "LIME":
                        with st.spinner("Calcul de LIME en cours..."):
                            # Appel de la nouvelle fonction
                            lime_result = get_lime(model, input_data)
                            st.image(lime_result, caption="Segments LIME (Top Features)", use_container_width=True)
                    
                    else:
                        st.image("https://via.placeholder.com/300?text=" + method, use_container_width=True)
else:
    st.info("Veuillez uploader un fichier (Audio .wav ou Image .jpg/.png) pour commencer.")