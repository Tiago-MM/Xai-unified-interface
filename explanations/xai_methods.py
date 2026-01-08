import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import cv2

def get_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Génère une heatmap Grad-CAM.
    Version robuste : gère les modèles standards (VGG) et imbriqués (DenseNet/Transfer Learning).
    """
    last_conv_layer_output = None
    grads = None
    
    # --- TENTATIVE 1 : Approche Standard (Pour VGG16, AlexNet, Custom CNN) ---
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
            
        grads = tape.gradient(class_channel, last_conv_layer_output)

    # --- TENTATIVE 2 : Fallback pour Modèles Imbriqués (DenseNet, ResNet via Transfer Learning) ---
    except (ValueError, Exception):
        # Si le graphe est déconnecté, on exécute le modèle en deux parties "manuellement"
        
        # 1. Récupération de la couche cible (le Backbone)
        target_layer = model.get_layer(last_conv_layer_name)
        
        # 2. Partie 1 : Extraction des features (Forward pass simple)
        # On passe l'image dans le backbone pour avoir les "cartes d'activation"
        conv_output = target_layer(img_array)
        
        # 3. Partie 2 : Calcul du gradient sur le reste du modèle (Classifieur)
        with tf.GradientTape() as tape:
            tape.watch(conv_output) # On surveille manuellement cette sortie intermédiaire
            x = conv_output
            
            # On cherche où se trouve notre couche cible dans la liste des couches
            start_index = 0
            for i, layer in enumerate(model.layers):
                if layer.name == last_conv_layer_name:
                    start_index = i + 1
                    break
            
            # On fait passer les features dans toutes les couches suivantes (Pooling, Dense, etc.)
            for layer in model.layers[start_index:]:
                x = layer(x)
            
            preds = x
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        # Calcul du gradient
        grads = tape.gradient(class_channel, conv_output)
        last_conv_layer_output = conv_output

    # --- GÉNÉRATION DE LA HEATMAP (Code Commun) ---
    
    # Moyenne des gradients (Importance des canaux)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiplication : Features * Poids d'importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisation ReLU et Min-Max
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    # Conversion explicite en numpy array pour éviter les erreurs de type plus loin
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
    
    input_img = img_array[0].astype('double')
    
    # --- FIX START: Gestion Grayscale vs RGB ---
    # Si l'image est en N&B (1 canal), on doit feinter LIME qui veut du RGB
    if input_img.shape[-1] == 1:
        # 1. On crée une fausse image RGB pour LIME (Stacking x3)
        # input_img[:,:,0] permet de passer de (224,224,1) à (224,224)
        image_to_explain = np.stack([input_img[:,:,0]]*3, axis=-1)
        
        # 2. On définit une fonction "wrapper" pour la prédiction
        # LIME va envoyer des images RGB (N, 224, 224, 3)
        # Nous on doit les remettre en (N, 224, 224, 1) pour le modèle
        def predict_wrapper(images):
            # Conversion RGB -> Grayscale par moyenne des canaux
            images_gray = np.mean(images, axis=-1, keepdims=True)
            return model.predict(images_gray)
            
        prediction_fn = predict_wrapper
    else:
        # Cas standard (VGG16, MobileNet, etc.) : Tout est déjà en RGB
        image_to_explain = input_img
        prediction_fn = model.predict
    # --- FIX END ---

    # explainer.explain_instance prend une image 3D (pas de batch)
    explanation = explainer.explain_instance(
        image_to_explain, 
        prediction_fn, # On utilise notre wrapper intelligent ou la fonction standard
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



import streamlit as st
import numpy as np
import shap
import tensorflow as tf

def explain_shap(model, input_data):
    # 1. Préparation du background (référence neutre)
    # On utilise une seule image noire de la même taille que l'entrée
    background = np.zeros_like(input_data) 
    input_shape = input_data.shape[1:] # (224, 224, 3) ou (224, 224, 1)

    def map_predict(x):
        # x arrive ici souvent avec des dimensions imprévues par KernelExplainer
        # On le reformate en 4D pour le modèle Keras
        reshaped_x = x.reshape((-1,) + input_shape)
        return model.predict(reshaped_x)

    # On aplatit l'entrée pour SHAP Kernel
    input_flat = input_data.reshape((1, -1))
    background_flat = background.reshape((1, -1))

    explainer = shap.KernelExplainer(map_predict, background_flat)
    
    # nsamples=50 est un compromis pour que Streamlit ne crash pas
    shap_values = explainer.shap_values(input_flat, nsamples=50)
    
    # On redimensionne le résultat pour qu'il retrouve sa forme d'image
    if isinstance(shap_values, list):
        # Pour chaque classe de sortie, on redimensionne
        return [sv.reshape(input_data.shape) for sv in shap_values]
    return shap_values.reshape(input_data.shape)