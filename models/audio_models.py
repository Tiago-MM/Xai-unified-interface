import tensorflow as tf
import tf_keras as keras
import os

# 2 directory of the script
script_dir = os.path.dirname(__file__)

def load_unified_model(model_name):
    # architecture detection and loading custom model
    if "VGG16" in model_name:
        return keras.models.load_model(os.path.join(script_dir, '..', '..', 'models', 'my_vgg16.h5'))
    elif "ResNet" in model_name:
        return keras.models.load_model(os.path.join(script_dir, '..', '..', 'models', 'my_resnet.h5'))
    elif "AlexNet" in model_name:
        return keras.models.load_model(os.path.join(script_dir, '..', '..', 'models', 'my_alexnet.h5'))
    elif "MobileNet" in model_name:
        return keras.models.load_model(os.path.join(script_dir, '..', '..', 'models', 'my_mobilenet.h5'))
    elif "DenseNet" in model_name:
        return keras.models.load_model(os.path.join(script_dir, '..', '..', 'models', 'my_densenet.h5'))
    
    return keras.applications.vgg16.VGG16(weights='imagenet')
