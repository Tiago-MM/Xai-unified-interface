import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def get_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
    """Génère une carte de chaleur Grad-CAM pour expliquer la prédiction."""
    # 1. Créer un modèle qui sort à la fois la couche conv et la prédiction
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Calculer le gradient pour la classe prédite
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Gradient de la classe par rapport à la sortie de la couche conv
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 4. Moyenne des gradients (poids des neurones)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Pondération de la sortie de la couche conv
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalisation de la heatmap pour l'affichage
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(heatmap, original_img):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # On s'assure que original_img est aussi en uint8 si ce n'est pas déjà le cas
    if original_img.max() <= 1.0:
        original_img = np.uint8(255 * original_img)
        
    superimposed_img = heatmap * 0.4 + original_img
    
    # On force le retour en uint8 (0-255) pour Streamlit
    return np.uint8(np.clip(superimposed_img, 0, 255))


from lime import lime_image
from skimage.segmentation import mark_boundaries

def get_lime(model, img_array):
    """Génère une explication LIME en identifiant les super-pixels influents."""
    explainer = lime_image.LimeImageExplainer()
    
    # explainer.explain_instance prend une image 3D (pas de batch)
    # On réduit num_samples à 50 pour la rapidité en démo
    explanation = explainer.explain_instance(
        img_array[0].astype('double'), 
        model.predict, 
        top_labels=1, 
        hide_color=0, 
        num_samples=50 
    )
    
    # Récupération du masque pour la classe prédite
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    # Transformation pour que l'image soit compatible avec Streamlit (0 à 1)
    img_boundaried = mark_boundaries(temp / temp.max(), mask)
    return img_boundaried