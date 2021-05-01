import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as pl
import numpy as np


class Predicter:
    MODEL_PATH = "models/model.h5"
    model = tf.keras.models.load_model(MODEL_PATH)

    def get_class_index_name(self, index):
        if index == 1:
            return "Covid не обнаружен"
        else:
            return "Covid обнаружен"

    def predict_image(self, image):
        img = np.expand_dims(image, axis=0)
        pred = self.model.predict(img)
        predClass = self.model.predict_classes(img)
        pred = np.around(pred, decimals=4, out=None)
        return self.get_class_index_name(predClass)
