import numpy as np
from keras.models import load_model

import config

model = None


def load():
    global model
    if model is None:
        model = load_model(config.MODEL_FILE_NAME)

def predict_is_ci(string):
    return get_ci_certainty(string) >= 0.5

def get_ci_certainty(string):
    global model
    if model is None:
        load()
    string = string[0:100].strip()
    while len(string) < 100:
        string += " "
    predicted = model((np.array([[ord(c) for c in string]])))
    predicted = predicted[0][0]
    return predicted