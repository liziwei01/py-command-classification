"""Module that allows interfacing with the model"""
import numpy as np
from keras.models import load_model

import config

model = None


def load():
    """Loads Model for Use."""
    global model
    if model is None:
        model = load_model(config.MODEL_FILE_NAME)

def predict_is_ci(string: str):
    """Predict if Command Injection

    Args:
        string (str): String to predict if it's command injection. Is automatically shortened/lengthened to 100 characters.

    Returns:
        tf.Tensor: A tensor containing a bool of True or False. Can be used in == compares, but not `is` compares.
    """
    return get_ci_certainty(string) >= 0.5

def get_ci_certainty(string: str):
    """Get Command Injection Certainty.

    Args:
        string (str): String to get certainty of command injection. Is automatically shortened/lengthened to 100 characters.

    Returns:
        float: Effectively the percentage chance that string is command injection
    """
    global model
    if model is None:
        load()
    string = string[0:100].strip()
    while len(string) < 100:
        string += " "
    predicted = model((np.array([[ord(c) for c in string]])))
    predicted = predicted[0][0]
    return predicted