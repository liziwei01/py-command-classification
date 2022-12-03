import os
import keras
from keras import layers, models
import numpy as np
import tensorflow as tf

# https://github.com/fchollet/deep-learning-with-python-notebooks/issues/157
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://keras.io/api/losses/probabilistic_losses/#categorical_crossentropy-function
# https://datascience.stackexchange.com/questions/85608/valueerror-input-0-of-layer-sequential-is-incompatible-with-the-layer-expected


training_data = []
training_labels = []
testing_data = []
testing_labels = []

for fil in os.listdir("data/train"):
	fil = os.path.join("data/train", fil)
	with open(fil, "r") as f:
		for line in f.readlines():
			if "is_command_injection" in fil:
				training_labels.append(np.array([1]))
			else:
				training_labels.append(np.array([0]))
			line = line.strip()
			while len(line) < 100:
				line += " "
			training_data.append(np.array([ord(c) for c in line[0:100]]))

for fil in os.listdir("data/test"):
	fil = os.path.join("data/test", fil)
	with open(fil, "r") as f:
		for line in f.readlines():
			if "is_command_injection" in fil:
				testing_labels.append(np.array([1]))
			else:
				testing_labels.append(np.array([0]))
			line = line.strip()
			while len(line) < 100:
				line += " "
			testing_data.append(np.array([ord(c) for c in line[0:100]]))

training_data = np.array(training_data)
testing_data = np.array(testing_data)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

if __name__ == "__main__":
	model = models.Sequential()
	model.add(layers.Dense(512, activation="relu", input_shape=(100,)))
	model.add(layers.Dense(128, activation="relu"))
	model.add(layers.Dense(32, activation="relu"))
	model.add(layers.Dense(1, activation="sigmoid"))

	model.compile(
		optimizer="Adam", loss="binary_crossentropy", metrics=["mae", "acc"]
	)

	model.fit(training_data, training_labels, epochs=200, batch_size=64, validation_data=(testing_data, testing_labels))

	model.save("model.h5")