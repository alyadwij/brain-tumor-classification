import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model_CNN = load_model("D:\\Kuliah\\PA\\PAgui\\Model\\model_do03dns320batch32kernel35353_LRsched+1clipnorm.keras", compile=False)
model_ResNet50 = load_model("D:\\Kuliah\\PA\\PAgui\\Model\\NEWresnet_model(3).keras", compile=False)
model_VGG16 = load_model("D:\\Kuliah\\PA\\PAgui\\Model\\NEWvgg16_model.keras", compile=False)
model_VGG19 = load_model("D:\\Kuliah\\PA\\PAgui\\Model\\NEWvgg19_model.keras", compile=False)

classes = ["glioma", "meningioma", "notumor", "pituitary"]

def predict(img, selected_model):
    if selected_model == "CNN":
        model = model_CNN
    elif selected_model == "ResNet50":
        model = model_ResNet50
    elif selected_model == "VGG-16":
        model = model_VGG16
    elif selected_model == "VGG-19":
        model = model_VGG19
    else:
        raise ValueError("Invalid model selection")
    
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    return classes[class_idx], confidence, pred