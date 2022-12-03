import train
import numpy as np
import prediction
import os

correct = 0
incorrect = 0

for i in range(len(train.testing_data_strings)):
    print("Testing {} of {}".format(i + 1, len(train.testing_data_strings)), end="\r")
    predicted = prediction.predict_is_ci(train.testing_data_strings[i])
    if predicted == (train.testing_labels[i][0] == 1):
        correct += 1
    else:
        incorrect += 1
    
print("\nCorrect: {}/{} ({}%)".format(correct, correct + incorrect, correct/(correct + incorrect) * 100))
    