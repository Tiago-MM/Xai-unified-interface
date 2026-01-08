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
import pandas as pd
import time

# Configuration de l'environnement pour éviter les crashs sur Mac
os.environ["TF_USE_LEGACY_KERAS"] = "1"
@st.cache_resource



def calculate_sparsity(heatmap):
    # Pourcentage de pixels avec une importance très faible (< 10% du max)
    return (np.sum(heatmap < 0.1 * heatmap.max()) / heatmap.size) * 100

def calculate_drop_score(model, input_data, heatmap):
    # Simule l'impact de la suppression des zones importantes (Fidélité)
    # Plus le score est haut, plus la zone expliquée est cruciale pour le modèle
    orig_pred = model.predict(input_data)[0].max()
    
    # On crée un masque : on garde les zones froides, on cache les zones chaudes
    mask = (heatmap < 0.5 * heatmap.max()).astype(float)
    mask = cv2.resize(mask, (input_data.shape[2], input_data.shape[1]))
    masked_input = input_data * np.expand_dims(np.stack([mask]*3, axis=-1), axis=0)
    
    new_pred = model.predict(masked_input)[0].max()
    drop = max(0, (orig_pred - new_pred) / orig_pred) * 100
    return drop

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
                    st.image(original_img, caption="Radiographie originale", width='stretch')
                with col2:
                    st.subheader("Explication Grad-CAM")
                    st.warning("Identification des zones tumorales suspectes dans les poumons.")


    with tab2:
        st.subheader("Analyse comparative Side-by-Side")
        
        if uploaded_file and len(selected_xai) >= 1:
            # Remplacer la ligne de sélection de conv_layer par celle-ci
            if "VGG16" in selected_model:
                conv_layer = "block5_conv3"
            elif "DenseNet" in selected_model:
                conv_layer = "conv5_block16_concat" # Nom extrait de votre erreur
            elif "AlexNet" in selected_model:
                conv_layer = "conv2d_4" # À vérifier selon votre modèle AlexNet
            else:
                # Par défaut, on prend la dernière couche avant le pooling si on ne connaît pas le nom
                conv_layer = [layer.name for layer in model.layers if isinstance(layer, (cv2.dnn_Layer, object)) and 'conv' in layer.name][-1]
            cols = st.columns(len(selected_xai))
            
            # Stockage pour le rapport d'audit
            metrics_data = []

            for i, method in enumerate(selected_xai):
                with cols[i]:
                    st.info(f"Méthode : {method}")
                    start_time = time.time()
                    
                    # --- GÉNÉRATION ---
                    if method == "Grad-CAM":
                        heatmap = get_gradcam(model, input_data, conv_layer)
                        base_img = cv2.resize(np.stack([S_db]*3, axis=-1) if input_type=="Audio" else np.array(original_img), (224,224))
                        result = superimpose_heatmap(heatmap, base_img)
                        st.image(result, caption="Heatmap Grad-CAM", use_container_width=True)
                        h_for_metric = heatmap

                    elif method == "LIME":
                        with st.spinner("Calcul de LIME..."):
                            lime_result = get_lime(model, input_data)
                            
                            # 1. Normalisation forcée pour éviter l'erreur
                            if lime_result.max() > 1.0:
                                # Si les valeurs sont déjà en 0-255 mais en float, on convertit juste
                                lime_display = lime_result.astype(np.uint8)
                            else:
                                # Si elles sont en 0-1, on les passe en 0-255
                                lime_display = (lime_result * 255).astype(np.uint8)
                            
                            # 2. Affichage avec l'image convertie
                            st.image(lime_display, caption="Segments LIME", use_container_width=True)
                            
                            # Utiliser l'image convertie pour les métriques de calcul
                            h_for_metric = cv2.cvtColor(lime_display, cv2.COLOR_RGB2GRAY)
                    
                    duration = time.time() - start_time

                    # --- CALCUL DES MÉTRIQUES ---
                    sparsity = calculate_sparsity(h_for_metric)
                    drop = calculate_drop_score(model, input_data, h_for_metric)
                    
                    metrics_data.append({
                        "Méthode": method,
                        "Sparsité (%)": sparsity,
                        "Fidélité (Drop %)": drop,
                        "Vitesse (s)": duration
                    })

            # --- AFFICHAGE DU RAPPORT D'AUDIT ---
            st.divider()
            st.subheader("📊 Rapport d'Audit Quantitatif")
            
            col_table, col_radar = st.columns([1, 1])
            
            df = pd.DataFrame(metrics_data)
            with col_table:
                st.dataframe(df.style.highlight_max(axis=0, subset=['Fidélité (Drop %)'], color='lightgreen'))

            with col_radar:
                st.write("📈 *Profil d'interprétabilité*")
                fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                
                categories = ['Fidélité', 'Sparsité', 'Vitesse']
                N = len(categories)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]

                for m in metrics_data:
                    # Normalisation pour le radar (0-100)
                    # Vitesse : plus c'est rapide (petit), plus le score est haut
                    v_score = max(0, 100 - (m['Vitesse (s)'] * 10)) 
                    values = [m['Fidélité (Drop %)'], m['Sparsité (%)'], v_score]
                    values += values[:1]
                    
                    ax.plot(angles, values, linewidth=2, label=m['Méthode'])
                    ax.fill(angles, values, alpha=0.1)

                plt.xticks(angles[:-1], categories)
                st.pyplot(fig)

            
else:
    st.info("Veuillez uploader un fichier (Audio .wav ou Image .jpg/.png) pour commencer.")