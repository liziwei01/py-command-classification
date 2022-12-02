'''
Author: liziwei01
Date: 2022-11-08 10:07:51
LastEditors: liziwei01
LastEditTime: 2022-12-02 10:52:55
Description: file content
'''
import glob
import os

import h5py
import numpy as np

TrainingDataDir = "data/train/"
TestingDataDir = "data/test/"
isCommandInjectionFile = "is_command_injection.txt"
notCommandInjectionFile = "not_command_injection.txt"
H5Dir = "data/h5/"
PreparedTrainingH5Name = "train"
PreparedTestingH5Name = "test"

SPECIAL_CHARS = []  # Used in line2words to split words

### configuration
inputSize = 13
###

'''
description: get special chars for spliting words
return {*}
'''
def initSpecialChars():
	global SPECIAL_CHARS
	for i in range(32):
		SPECIAL_CHARS.append(chr(i))  # Add control characters
	SPECIAL_CHARS += [char for char in "~`#$&*()\\|[]}{;'\"<>/!?\n\nt"]  # Add bash special characters

'''
description: `pwd`
param {str} fileName
return {str}
'''
def GetFullPath(fileName):
	return os.path.join(os.getcwd(), fileName)

'''
description: `find $dataset -name "*.txt"` 
param {str} dataset
return {list}
'''
def GetFilePaths(dataset=TrainingDataDir):
	dataset = GetFullPath(dataset)
	data = glob.glob(os.path.join(dataset, "*.txt"))
	return data

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
	saved_path = os.path.join(H5Dir, file_name.lower()+".h5")
	with h5py.File(saved_path, 'w') as hf:
		hf.create_dataset('data', data=arr_data)
		hf.create_dataset('label', data=arr_label)

'''
description: get inputs and labels
param {str} file_name
return {tuple[list, list]}
'''
def GetH5File(file_name="train"):
	data_dir = os.path.join(H5Dir, file_name.lower()+".h5")
	with h5py.File(data_dir, "r") as hf:
		inputs = np.array(hf.get("data"))
		labels = np.array(hf.get("label"))
		return inputs, labels

'''
description: convert training txt file to h5
return {*}
'''
def prepareTrainingData():
	# lines of commands
	sub_input_sequence = []
	# label is 1 if this line of input is command injection, 0 otherwise
	sub_label_sequence = []
	file_paths = GetFilePaths(TrainingDataDir)

	for file_path in file_paths:
		with open(file_path, "r") as f:
			lines = f.readlines()
			for line in lines:
				sub_input = getInput(line)
				sub_label = np.array([0]).reshape([1, 1, 1])
				if isCommandInjectionFile in file_path:
					sub_label = np.array([1]).reshape([1, 1, 1])
				sub_input_sequence.append(sub_input)
				sub_label_sequence.append(sub_label)
	saveAsPreparedH5(sub_input_sequence, sub_label_sequence, PreparedTrainingH5Name)

'''
description: convert testing txt file to h5
return {*}
'''
def prepareTestingData():
	sub_input_sequence = []
	sub_label_sequence = []
	file_paths = GetFilePaths(TestingDataDir)

	for file_path in file_paths:
		with open(file_path, "r") as f:
			lines = f.readlines()
			for line in lines:
				sub_input = getInput(line)
				sub_label = np.array([0]).reshape([1, 1, 1])
				if isCommandInjectionFile in file_path:
					sub_label = np.array([1]).reshape([1, 1, 1])
				sub_input_sequence.append(sub_input)
				sub_label_sequence.append(sub_label)
	saveAsPreparedH5(sub_input_sequence, sub_label_sequence, PreparedTestingH5Name)


'''
description: normalize the input matrix
param {str} line
return {*}
'''
def getInput(line):
	res = command2Matrix(line).reshape([inputSize, inputSize, 1])
	ret = np.around(res / 128., decimals=4)
	return ret

'''
description: 
param {str} command
return {list}
'''
def command2Matrix(command):
	res = np.arange(0, 13).reshape([1, 13])
	# divide line to words
	words = line2words(command)
	# add placeholder
	for i in range(inputSize):
		words.append([0])
	for i in range(len(words)):
		for j in range(inputSize):
			words[i].append(0)
		words[i] = words[i][:inputSize]

	for i in range(inputSize+1):
		inArr = np.array(words[i]).reshape([1, inputSize])
		res = np.append(res, inArr, axis=0)
	
	ret = res[1:inputSize+1]
	return ret

'''
description: divide command into words by bash characters and ASCII control codes
param {str} command
return {list}
'''
def line2words(command):
	line = []
	lines = []
	for i in range(len(command)):
		if i == len(command)-1:
			lines.append(line)
		if command[i] in SPECIAL_CHARS:
			if len(line) != 0:
				lines.append(line)
				line = []
		else:
			char = ord(command[i])
			line.append(char)
	return lines


def Prepare():
	# prepareTrainingData()
	prepareTestingData()


if __name__ == "__main__":
	initSpecialChars()
	Prepare()

