import numpy as np
import cv2

def shap_to_image(shap_values, original_img):
    # shap list handling
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # delete batch dimension if present
    if len(shap_values.shape) == 4:
        shap_values = shap_values[0]

    # mean for heatmap
    shap_img = np.abs(shap_values).sum(axis=-1)
    
    # Normalisation 
    shap_min, shap_max = shap_img.min(), shap_img.max()
    if shap_max > shap_min:
        shap_img = (shap_img - shap_min) / (shap_max - shap_min)
    
    # Handle resizing
    shap_img_resized = cv2.resize(shap_img, (original_img.shape[1], original_img.shape[0]))
    
    # color map
    heatmap = np.uint8(255 * shap_img_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    if original_img.max() <= 1.0:
        background = np.uint8(255 * original_img)
    else:
        background = original_img.astype(np.uint8)
    
    # superimpose
    return cv2.addWeighted(background, 0.6, heatmap, 0.4, 0)

def calculate_sparsity(heatmap):
    # return percentage of pixels below 10% of max value( ie, cold areas, non important)
    return (np.sum(heatmap < 0.1 * heatmap.max()) / heatmap.size) * 100

def calculate_drop_score(model, input_data, heatmap):
    # original prediction
    orig_pred = model.predict(input_data)[0].max()
    
    # 1.creation of the mask with only cold areas
    mask = (heatmap < 0.5 * heatmap.max()).astype(float)
    
    # size mask to input size
    mask = cv2.resize(mask, (input_data.shape[2], input_data.shape[1]))
    
    nb_channels = input_data.shape[-1]  
    mask_stacked = np.stack([mask] * nb_channels, axis=-1)
    
    # dimension batch
    mask_expanded = np.expand_dims(mask_stacked, axis=0)
    
    # application with mask
    masked_input = input_data * mask_expanded
    
    new_pred = model.predict(masked_input)[0].max()
    drop = max(0, (orig_pred - new_pred) / orig_pred) * 100
    return drop


