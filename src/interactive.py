from keras.models import load_model
import train
import numpy as np
import prediction

if __name__ == "__main__":
    while True:
        inp = input("Enter string: ")
        is_ci = prediction.predict_is_ci(inp)
        out = "Prediction: '{}' Is {}Command Injection!".format(inp, "" if is_ci else "Not ")
        print(out)
    