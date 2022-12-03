'''
Author: bfishe32
Date: 2022-12-03 00:47:09
LastEditors: liziwei01
LastEditTime: 2022-12-03 11:36:10
Description: file content
'''
import os
import numpy as np
import h5py
from config import config

'''
description: save as h5 for faster reading
param {list} sub_input_sequence
param {list} sub_label_sequence
param {str} file_name
return {*}
'''
def saveAsPreparedH5(sub_input_sequence, sub_label_sequence, file_name="train"):
	arr_data = np.asarray(sub_input_sequence)
	arr_label = np.asarray(sub_label_sequence)
	saved_path = os.path.join(config.H5Dir, file_name.lower()+".h5")
	with h5py.File(saved_path, 'w') as hf:
		hf.create_dataset("data", data=arr_data)
		hf.create_dataset("labels", data=arr_label)

'''
description: get inputs and labels
param {str} file_name
return {tuple[list, list]}
'''
def GetH5File(file_name="train"):
	data_dir = os.path.join(config.H5Dir, file_name.lower()+".h5")
	with h5py.File(data_dir, "r") as hf:
		inputs = np.array(hf.get("data"))
		labels = np.array(hf.get("labels"))
		return inputs, labels

'''
description: convert training txt file to h5
return {*}
'''
def prepareTrainingData():
	training_data = []
	training_labels = []
	for fil in os.listdir(config.TrainDir):
		fil = os.path.join(config.TrainDir, fil)
		with open(fil, "r") as f:
			for line in f.readlines():
				if config.CommandInjectionFileName in fil:
					training_labels.append(np.array([1]))
				else:
					training_labels.append(np.array([0]))
				line = line.strip()
				while len(line) < 100:
					line += " "
				training_data.append(np.array([ord(c) for c in line[0:100]]))
	saveAsPreparedH5(training_data, training_labels, config.TrainH5FileName)


'''
description: convert testing txt file to h5
return {*}
'''
def prepareTestingData():
	testing_data = []
	testing_labels = []
	for fil in os.listdir(config.TestDir):
		fil = os.path.join(config.TestDir, fil)
		with open(fil, "r") as f:
			for line in f.readlines():
				if config.CommandInjectionFileName in fil:
					testing_labels.append(np.array([1]))
				else:
					testing_labels.append(np.array([0]))
				line = line.strip()
				while len(line) < 100:
					line += " "
				testing_data.append(np.array([ord(c) for c in line[0:100]]))
	saveAsPreparedH5(testing_data, testing_labels, config.TestH5FileName)

if __name__ == "__main__":
	prepareTrainingData()
	prepareTestingData()