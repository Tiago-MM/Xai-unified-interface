import numpy as np
import cv2
from PIL import Image

def process_chest_xray(uploaded_file):
    # read
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    # resize
    img_resized = cv2.resize(img_array, (224, 224))
    img_final = np.expand_dims(img_resized, axis=0)
    # normalize
    img_final = img_final / 255.0
    return img_final, image