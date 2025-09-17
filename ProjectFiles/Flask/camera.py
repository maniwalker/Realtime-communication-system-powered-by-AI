import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class Predictor:
    def __init__(self):
		base_dir = os.path.dirname(os.path.abspath(__file__))
        zip_path = os.path.join(base_dir, "asl_model.zip")   
        model_path = os.path.join(base_dir, "asl_model.h5") 
        # Extract the model if it doesn't exist
        if not os.path.exists(model_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            	zip_ref.extractall(base_dir)
        self.model = load_model(model_path)
        self.index = ['A','B','C','D','E','F','G','H','I']

    def predict_frame(self, frame):
        roi = frame[150:350, 50:250]
        roi_resized = cv2.resize(roi, (64, 64))

        x = image.img_to_array(roi_resized)
        x = np.expand_dims(x, axis=0)

        pred = np.argmax(self.model.predict(x), axis=1)[0]
        return self.index[pred]
