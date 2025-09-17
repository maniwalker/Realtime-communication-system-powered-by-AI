import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class Predictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "asl_model.h5")
        self.model = load_model(model_path)
        self.index = ['A','B','C','D','E','F','G','H','I']

    def predict_frame(self, frame):
        # Extract ROI
        roi = frame[150:350, 50:250]
        roi_resized = cv2.resize(roi, (64, 64))

        x = image.img_to_array(roi_resized)
        x = np.expand_dims(x, axis=0)

        pred = np.argmax(self.model.predict(x), axis=1)[0]
        return self.index[pred]
