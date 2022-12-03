from keras.models import load_model
import train
import numpy as np

model = load_model("test.h5")

correct = 0
incorrect = 0

def is_correct(predicted, actual):
    actual = actual[0]
    predicted = predicted[0][0]
    if actual == 0:
        return predicted < 0.5
    else:
        return predicted >= 0.5

for i in range(len(train.testing_data)):
    print("Testing {} of {}".format(i + 1, len(train.testing_data)), end="\r")
    prediction = model((np.array([train.testing_data[i]])))
    if is_correct(prediction, train.testing_labels[i]):
        correct += 1
    else:
        incorrect += 1
    
print("\nCorrect: {}/{} ({}%)".format(correct, correct + incorrect, correct/(correct + incorrect) * 100))
    