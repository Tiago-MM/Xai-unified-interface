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
from explanations.xai_methods import get_gradcam, superimpose_heatmap, get_lime, explain_shap
from utils.audio_proc import process_audio
import pandas as pd
import time

# Configuration de l'environnement pour éviter les crashs sur Mac
os.environ["TF_USE_LEGACY_KERAS"] = "1"
@st.cache_resource


def shap_to_image(shap_values, original_img):
    # SHAP retourne souvent une liste pour chaque classe. 
    # On prend les valeurs pour la classe prédite (souvent l'index 0 ou 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Suppression des dimensions inutiles (ex: batch size 1)
    if len(shap_values.shape) == 4:
        shap_values = shap_values[0]

    # Moyenne sur les canaux (RGB -> Grayscale) pour la heatmap
    shap_img = np.abs(shap_values).sum(axis=-1)
    
    # Normalisation Min-Max robuste
    shap_min, shap_max = shap_img.min(), shap_img.max()
    if shap_max > shap_min:
        shap_img = (shap_img - shap_min) / (shap_max - shap_min)
    
    # Redimensionnement pour correspondre à l'image originale
    shap_img_resized = cv2.resize(shap_img, (original_img.shape[1], original_img.shape[0]))
    
    # Création de la heatmap colorée
    heatmap = np.uint8(255 * shap_img_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Préparation de l'image de fond
    if original_img.max() <= 1.0:
        background = np.uint8(255 * original_img)
    else:
        background = original_img.astype(np.uint8)
    
    # Superposition
    return cv2.addWeighted(background, 0.6, heatmap, 0.4, 0)

def calculate_sparsity(heatmap):
    # Pourcentage de pixels avec une importance très faible (< 10% du max)
    return (np.sum(heatmap < 0.1 * heatmap.max()) / heatmap.size) * 100

def calculate_drop_score(model, input_data, heatmap):
    # Simule l'impact de la suppression des zones importantes (Fidélité)
    orig_pred = model.predict(input_data)[0].max()
    
    # 1. Création du masque (zones froides conservées)
    mask = (heatmap < 0.5 * heatmap.max()).astype(float)
    
    # 2. Redimensionnement du masque à la taille de l'image (224x224)
    mask = cv2.resize(mask, (input_data.shape[2], input_data.shape[1]))
    
    # 3. CORRECTION : Adaptation automatique aux canaux (1 ou 3)
    # On récupère le nombre de canaux de l'entrée actuelle (1 pour AlexNet, 3 pour VGG)
    nb_channels = input_data.shape[-1] 
    
    # On empile le masque autant de fois que nécessaire (x1 ou x3)
    mask_stacked = np.stack([mask] * nb_channels, axis=-1)
    
    # On ajoute la dimension batch (1, 224, 224, C)
    mask_expanded = np.expand_dims(mask_stacked, axis=0)
    
    # 4. Application du masque
    masked_input = input_data * mask_expanded
    
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


try :
        
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
                    
                    col1 = st.columns(1)[0]
                    with col1:
                        st.metric("Verdict Audio", label, f"Score: {score*100:.2f}%")
                        fig, ax = plt.subplots()
                        librosa.display.specshow(S_db, ax=ax)
                        st.pyplot(fig)

                else: # Traitement IMAGE (Repo 2)
                    input_data, original_img = process_chest_xray(uploaded_file)
                    if model.input_shape[-1] == 1 and input_data.shape[-1] == 3:
                        # Conversion moyenne des canaux pour passer de (1, 224, 224, 3) à (1, 224, 224, 1)
                        input_data = np.mean(input_data, axis=-1, keepdims=True)
                    preds = model.predict(input_data)
                    score = np.max(preds[0])
                    label = "MALIGNANT" if score > 0.5 else "BENIGN / NORMAL"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Verdict Médical", label, f"Score: {score*100:.2f}%")
                        st.image(original_img, caption="Radiographie originale", width='stretch')
                    with col2:
                        st.subheader("Explication Grad-CAM")
                        st.warning("Identification des zones tumorales suspectes dans les poumons.")


        with tab2:
            st.subheader("Analyse comparative Side-by-Side")
                        
            st.write("---")
            st.subheader("📍 Positionnement du Verdict")

            # Création d'une barre de progression (0.0 à 1.0)
            st.progress(float(score))

            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_left:
                st.caption("✅ **0%** (Authentique / Sain)")
            with col_right:
                st.caption("🚨 **100%** (Fake / Cancer)")

            # Rappel de la règle de décision
            if score > 0.5:
                st.warning(f"Le score de **{score*100:.1f}%** dépasse le seuil de risque de 50%.")
            else:
                st.success(f"Le score de **{score*100:.1f}%** est inférieur au seuil de risque.")
            
            st.write("---")

            
            if uploaded_file and len(selected_xai) >= 1:
                # densenet only conv layer name fix
                if "DenseNet" in selected_model:
                    conv_layer = "densenet121" # Nom extrait de  erreur
                else:
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
                        
                        elif method == "SHAP":
                            with st.spinner("Calcul de SHAP..."):
                                shap_vals = explain_shap(model, input_data)
                                base_img = np.array(original_img) if input_type=="Image" else np.stack([S_db]*3, axis=-1)
                                
                                result = shap_to_image(shap_vals, base_img)
                                st.image(result, caption="Heatmap SHAP", use_container_width=True)
                                
                                # --- CORRECTION ICI ---
                                if isinstance(shap_vals, list): shap_vals = shap_vals[0]
                                if len(shap_vals.shape) == 4: shap_vals = shap_vals[0]
                                
                                # On garde une carte 2D (H, W) pour le calcul des métriques
                                h_for_metric = np.abs(shap_vals).sum(axis=-1)
                        
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
                
                st.write("---")
                with st.expander("ℹ️ Comment interpréter les résultats ?"):
                    st.markdown("""
                    Ce tableau vous aide à croiser le **Verdict (Confiance)** avec la **Preuve visuelle (Fidélité/Drop Score)**. 
                    
                    *Le score de confiance est ici exprimé par rapport au risque (0% = Sain/Vrai, 100% = Malade/Fake).*

                    | Cas de figure | Confiance (Score IA) | Fidélité (Drop Score) | Interprétation |
                    | :--- | :--- | :--- | :--- |
                    | **1. Diagnostic Robuste** | 🟢 **Élevée** (> 80%) | 🟢 **Élevée** (> 50%) | ✅ **Fiable.** Le modèle est sûr de lui et la zone rouge est la cause directe de sa décision. |
                    | **2. Le "Bluffeur" (Biais)** | 🟢 **Élevée** (> 80%) | 🔴 **Faible** (< 20%) | ⚠️ **Méfiance.** Le modèle est sûr, mais il regarde ailleurs (biais, bruit de fond, logo). |
                    | **3. Le Signal Faible** | 🟠 **Moyenne** (50-80%) | 🟢 **Élevée** (> 50%) | 🔸 **Investiguer.** Le modèle hésite, mais il a repéré une anomalie très précise dans la zone rouge. |
                    | **4. Suspicion Levée** | 🔴 **Faible** (< 20%) | 🟢 **Élevée** (> 50%) | ✅ **Rassurant.** Le modèle a analysé la zone suspecte mais a conclu qu'elle était normale. |
                    | **5. L'Aléatoire** | 🔴 **Faible** (< 50%) | 🔴 **Faible** (< 20%) | ❌ **Rejet.** Le modèle ne sait pas et l'explication est floue (bruit). |
                    """)

                
    else:
        st.info("Veuillez uploader un fichier (Audio .wav ou Image .jpg/.png) pour commencer.")

except Exception as e:
    st.error(f" Loading .... : {e}")