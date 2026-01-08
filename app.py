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
from utils.image_proc import process_chest_xray
from utils.functions_utils import shap_to_image, calculate_sparsity, calculate_drop_score
import pandas as pd
import time

# avoid beug
os.environ["TF_USE_LEGACY_KERAS"] = "1"
@st.cache_resource



def main():
    return ''

try :
        
    # streamlit config
    st.set_page_config(page_title="XAI Unified Interface", layout="wide")
    st.title("🛡️ Unified Explainable AI Interface")

    st.sidebar.header("data input & model selection")
    uploaded_file = st.sidebar.file_uploader("Drag & drop a file", type=['wav', 'png', 'jpg', 'jpeg'])

    if uploaded_file:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # verify file type
        if file_ext == 'wav':
            input_type = "Audio"
            available_models = ["VGG16 (Audio)", "ResNet", "MobileNet"]
            available_xai = ["Grad-CAM", "LIME", "SHAP"] 
        else:
            input_type = "Image"
            available_models = ["AlexNet (Cancer)", "DenseNet"]
            available_xai = ["Grad-CAM", "LIME", "SHAP"] 

        selected_model = st.sidebar.selectbox("Classification models", available_models)
        selected_xai = st.sidebar.multiselect("Xai Methods", available_xai) 

        tab1, tab2 = st.tabs(["🚀 Analysis", "📊 Explainability"]) 
        input_data = None
        model = None
        with tab1:
            if st.button("Launch Analysis"):
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

                else: # Image
                    input_data, original_img = process_chest_xray(uploaded_file)
                    if model.input_shape[-1] == 1 and input_data.shape[-1] == 3:
                        
                        input_data = np.mean(input_data, axis=-1, keepdims=True)
                    preds = model.predict(input_data)
                    score = np.max(preds[0])
                    label = "MALIGNANT" if score > 0.5 else "BENIGN / NORMAL"
                    
                    col1 = st.columns(1)[0]
                    with col1:
                        st.metric("Medical score", label, f"Score: {score*100:.2f}%")
                        st.image(original_img, caption="Radiographie originale", width='stretch')


        with tab2:
            st.subheader("Analysis Summary")
                        
            st.write("---")
            st.subheader("📍Position")

            # Display progress bar as score indicator
            st.progress(float(score))

            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_left:
                st.caption("✅ **0%** (Authentic / Benign)")
            with col_right:
                st.caption("🚨 **100%** (Fake / Cancer)")

            if score > 0.5:
                st.warning(f"The score **{score*100:.1f}%** exceeds the risk threshold of 50%.")
            else:
                st.success(f"The score of **{score*100:.1f}%** is below the risk threshold.")
            
            st.write("---")

            
            if uploaded_file and len(selected_xai) >= 1:
                # densenet only conv layer name fix
                if "DenseNet" in selected_model:
                    conv_layer = "densenet121" 
                else:
                    conv_layer = [layer.name for layer in model.layers if isinstance(layer, (cv2.dnn_Layer, object)) and 'conv' in layer.name][-1]
                cols = st.columns(len(selected_xai))
                
                # Audit metrics storage
                metrics_data = []

                for i, method in enumerate(selected_xai):
                    with cols[i]:
                        st.info(f"Méthode : {method}")
                        start_time = time.time()
                        
                        # Generate explanation
                        if method == "Grad-CAM":
                            heatmap = get_gradcam(model, input_data, conv_layer)
                            base_img = cv2.resize(np.stack([S_db]*3, axis=-1) if input_type=="Audio" else np.array(original_img), (224,224))
                            result = superimpose_heatmap(heatmap, base_img)
                            st.image(result, caption="Heatmap Grad-CAM", use_container_width=True)
                            h_for_metric = heatmap

                        elif method == "LIME":
                            with st.spinner("Calcul de LIME..."):
                                lime_result = get_lime(model, input_data)
                                
                                if lime_result.max() > 1.0:
                                    lime_display = lime_result.astype(np.uint8)
                                else:
                                    lime_display = (lime_result * 255).astype(np.uint8)
                                
                                st.image(lime_display, caption="Segments LIME", use_container_width=True)
                                
                                h_for_metric = cv2.cvtColor(lime_display, cv2.COLOR_RGB2GRAY)
                        
                        elif method == "SHAP":
                            with st.spinner("Calcul de SHAP..."):
                                shap_vals = explain_shap(model, input_data)
                                base_img = np.array(original_img) if input_type=="Image" else np.stack([S_db]*3, axis=-1)
                                
                                result = shap_to_image(shap_vals, base_img)
                                st.image(result, caption="Heatmap SHAP", use_container_width=True)
                                
                                if isinstance(shap_vals, list): shap_vals = shap_vals[0]
                                if len(shap_vals.shape) == 4: shap_vals = shap_vals[0]
                                
                                h_for_metric = np.abs(shap_vals).sum(axis=-1)
                        
                        duration = time.time() - start_time

                        # metrics calculation
                        sparsity = calculate_sparsity(h_for_metric)
                        drop = calculate_drop_score(model, input_data, h_for_metric)
                        
                        metrics_data.append({
                            "Méthode": method,
                            "Sparsity (%)": sparsity,
                            "Fidelity (Drop %)": drop,
                            "Latency (s)": duration
                        })

                # display audit report
                st.divider()
                st.subheader("📊 Audit Quantitatif résumé")
                
                col_table, col_radar = st.columns([1, 1])
                
                df = pd.DataFrame(metrics_data)
                with col_table:
                    st.dataframe(df.style.highlight_max(axis=0, subset=['Fidelity (Drop %)'], color='lightgreen'))

                with col_radar:
                    st.write("📈 *Interpretability Profile*")
                    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                    
                    categories = ['Fidelity', 'Sparsity', 'Latency']
                    N = len(categories)
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]

                    for m in metrics_data:
                        v_score = max(0, 100 - (m['Latency (s)'] * 10)) 
                        values = [m['Fidelity (Drop %)'], m['Sparsity (%)'], v_score]
                        values += values[:1]
                        
                        ax.plot(angles, values, linewidth=2, label=m['Méthode'])
                        ax.fill(angles, values, alpha=0.1)

                    plt.xticks(angles[:-1], categories)
                    st.pyplot(fig)
                
                st.write("---")
                with st.expander("ℹ️ How to interpret the results?"):
                    st.markdown("""
                    This table helps you correlate the **Verdict (Confidence)** with the **Visual Evidence (Fidelity/Drop Score)**.
                    
                    *Note: The confidence score represents the risk level (0% = Authentic/Healthy, 100% = Deepfake/Malignant).*

                    | Scenario | Confidence (AI Score) | Fidelity (Drop Score) | Interpretation |
                    | :--- | :--- | :--- | :--- |
                    | **1. Robust Diagnosis** | 🟢 **High** (> 80%) | 🟢 **High** (> 50%) | ✅ **Reliable.** Strong evidence of an anomaly found in the highlighted area. |
                    | **2. The "Bluffer" (Bias)** | 🟢 **High** (> 80%) | 🔴 **Low** (< 20%) | ⚠️ **Caution.** Likely bias; the model is confident but looking at irrelevant areas. |
                    | **3. Weak Signal** | 🟠 **Medium** (50-80%) | 🟢 **High** (> 50%) | 🔸 **Investigate.** Potential early detection or subtle anomaly identified. |
                    | **4. Suspicion Cleared** | 🔵 **Low** (< 20%) | 🟢 **High** (> 50%) | ✅ **Reassuring.** A suspicious area was analyzed and confirmed as normal/healthy. |
                    | **5. Random / Noise** | 🟠 **Medium** (~ 50%) | 🔴 **Low** (< 20%) | ❌ **Inconclusive.** The model is confused and the explanation is unstable. |
                    | **6. No Signal** | 🔵 **Null** (~ 0%) | ⚪ **Null** (0%) | ✅ **Safe.** No suspicious features or traces were found at all. |
                    """)
                
    else:
        st.info("Please upload an audio (.wav) or image (.png, .jpg) file to begin the analysis.")

except Exception as e:
    st.error(f" Loading .... : {e}")