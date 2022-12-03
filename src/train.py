'''
Author: bfishe32
Date: 2022-12-03 00:46:19
LastEditors: liziwei01
LastEditTime: 2022-12-03 11:41:06
Description: file content
'''
import prepare
from keras import layers, models, callbacks
from config import config
from os import path
import time

# https://github.com/fchollet/deep-learning-with-python-notebooks/issues/157
# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# https://keras.io/api/losses/probabilistic_losses/#categorical_crossentropy-function
# https://datascience.stackexchange.com/questions/85608/valueerror-input-0-of-layer-sequential-is-incompatible-with-the-layer-expected


def train():
	# The thought process here is to first expand to a massive amount of nodes to get as much data as possible, then
	# shrink down until we only have one node. From there, binary_crossentropy as a loss function will
	# help us to collapse to 0 or 1, which denote not command injection and command injection respectively.
	training_data, training_labels = prepare.GetH5File(config.TrainH5FileName)
	testing_data, testing_labels = prepare.GetH5File(config.TestH5FileName)
	start_time = time.time()
	print("Training...")
	for ep in range(config.Epoch):
		model = models.Sequential()
		model_file = path.join(config.H5Dir, config.ModelFileName+".h5")
		callbacks_list = []

		model.add(layers.Dense(512, activation="relu", input_shape=(100,)))
		model.add(layers.Dense(128, activation="relu"))
		model.add(layers.Dense(32, activation="relu"))
		model.add(layers.Dense(1, activation="sigmoid"))
		model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["mae", "acc"])
		if path.exists(model_file):
			checkpoint = callbacks.ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True, mode='min')
			callbacks_list = [checkpoint]
			print("Continuing...")
			model.fit(training_data, training_labels, epochs=1000, batch_size=64, validation_data=(testing_data, testing_labels), callbacks=callbacks_list)
		else:
			model.fit(training_data, training_labels, epochs=100, batch_size=64, validation_data=(testing_data, testing_labels))
		model.save(model_file)
		print("Epoch: [%2d], time: [%4.4f]" % ((ep+1), time.time()-start_time))

# Run training if from __main__
if __name__ == "__main__":
	train()