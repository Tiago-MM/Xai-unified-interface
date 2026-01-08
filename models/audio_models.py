import tensorflow as tf
import tf_keras as keras

def load_unified_model(model_name):
    # Les architectures pour Repo 1 (Audio) et Repo 2 (Image)
    if "VGG16" in model_name:
        return keras.models.load_model('models/my_vgg16.h5')
    elif "ResNet" in model_name:
        return keras.models.load_model('models/my_resnet.h5')
    elif "AlexNet" in model_name:
        return keras.applications.resnet50.ResNet50(weights='imagenet')
    elif "MobileNet" in model_name:
        return keras.models.load_model('models/my_mobilenet.h5')
    elif "DenseNet" in model_name:
        return keras.applications.densenet.DenseNet121(weights='imagenet')
    return keras.applications.vgg16.VGG16(weights='imagenet')