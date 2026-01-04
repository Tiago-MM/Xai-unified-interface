import tensorflow as tf
import tf_keras as keras

def load_unified_model(model_name):
    # Les architectures pour Repo 1 (Audio) et Repo 2 (Image)
    if "VGG16" in model_name:
        return keras.applications.vgg16.VGG16(weights='imagenet')
    elif "ResNet" in model_name or "AlexNet" in model_name:
        return keras.applications.resnet50.ResNet50(weights='imagenet')
    elif "MobileNet" in model_name or "DenseNet" in model_name:
        return keras.applications.densenet.DenseNet121(weights='imagenet')
    return keras.applications.vgg16.VGG16(weights='imagenet')