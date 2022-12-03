from keras.models import load_model
import train
import numpy as np

model = load_model("test.h5")


def get_prediction_str(predicted):
    predicted = predicted[0][0]
    if predicted < 0.5:
        return "Not a Command Injection!"
    return "Command Injection!"

while True:
    inp = input("Enter string: ")
    inp = inp[0:100].strip()
    while len(inp) < 100:
        inp += " "
    prediction = model((np.array([[ord(c) for c in inp]])))
    print(get_prediction_str(prediction))
    