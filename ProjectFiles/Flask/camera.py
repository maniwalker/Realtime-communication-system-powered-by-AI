import cv2
import numpy as np
import os
import zipfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class Predictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        zip_path = os.path.join(base_dir, "asl_model.zip")
        model_path = os.path.join(base_dir, "asl_model.h5")

        # Extract the model if it doesn't exist
        if not os.path.exists(model_path) and os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_dir)

        self.model = load_model(model_path)
        self.index = ['A','B','C','D','E','F','G','H','I']

    def predict_frame(self, frame):
        try:
            # Extract ROI safely
            roi = frame[150:350, 50:250]
            if roi.size == 0:
                return None

            # Preprocess
            roi_resized = cv2.resize(roi, (64, 64))
            x = image.img_to_array(roi_resized) / 255.0  # normalize
            x = np.expand_dims(x, axis=0)

            # Prediction
            pred = self.model.predict(x, verbose=0)  # suppress logs
            label = self.index[np.argmax(pred)]
            return label

        except Exception as e:
            print(f"Prediction error: {e}")
            return None
