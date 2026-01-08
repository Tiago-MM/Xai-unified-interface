import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries

# gradcam explanation
def get_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
    last_conv_layer_output = None
    grads = None
    
    # use for VGG16, AlexNet, MobileNet, but fails (sometimes) for DenseNet, ResNet
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

    # Case for DenseNet, ResNet 
    except (ValueError, Exception):
        target_layer = model.get_layer(last_conv_layer_name)
        # activation of the target layer
        conv_output = target_layer(img_array)
        
        # gradient calculation
        with tf.GradientTape() as tape:
            tape.watch(conv_output) 
            x = conv_output
            
            start_index = 0
            for i, layer in enumerate(model.layers):
                if layer.name == last_conv_layer_name:
                    start_index = i + 1
                    break
            
            for layer in model.layers[start_index:]:
                x = layer(x)
            
            preds = x
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_output)
        last_conv_layer_output = conv_output



    # mean grads
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # multiply each channel by "importance"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisation ReLU and Min-Max
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

# superimpose heatmap
def superimpose_heatmap(heatmap, original_img):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # same type conversion
    if original_img.max() <= 1.0:
        original_img = np.uint8(255 * original_img)
        
    superimposed_img = heatmap * 0.4 + original_img
    
    return np.uint8(np.clip(superimposed_img, 0, 255))



# lime explanation
def get_lime(model, img_array):
    explainer = lime_image.LimeImageExplainer()
    
    input_img = img_array[0].astype('double')
    
    # case black and white images
    if input_img.shape[-1] == 1:
        # rgb conversion
        image_to_explain = np.stack([input_img[:,:,0]]*3, axis=-1)
        def predict_wrapper(images):
            images_gray = np.mean(images, axis=-1, keepdims=True)
            return model.predict(images_gray)
            
        prediction_fn = predict_wrapper
    else:
        # case rgb images
        image_to_explain = input_img
        prediction_fn = model.predict

    # explanation
    explanation = explainer.explain_instance(
        image_to_explain, 
        prediction_fn,
        top_labels=1, 
        hide_color=0, 
        num_samples=50 
    )
    
    # mask and image
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    # image for display streamlit
    img_boundaried = mark_boundaries(temp / temp.max(), mask)
    return img_boundaried



import streamlit as st
import numpy as np
import shap
import tensorflow as tf

def explain_shap(model, input_data):
    # background for SHAP
    background = np.zeros_like(input_data) 
    input_shape = input_data.shape[1:] 

    # use for mapping predict function
    def map_predict(x):
        reshaped_x = x.reshape((-1,) + input_shape)
        return model.predict(reshaped_x)

    # reshape input and background
    input_flat = input_data.reshape((1, -1))
    background_flat = background.reshape((1, -1))

    explainer = shap.KernelExplainer(map_predict, background_flat)
    
    shap_values = explainer.shap_values(input_flat, nsamples=50) # more is better but slower
    
    # resize shap values to input shape
    if isinstance(shap_values, list):
        return [sv.reshape(input_data.shape) for sv in shap_values]
    return shap_values.reshape(input_data.shape)