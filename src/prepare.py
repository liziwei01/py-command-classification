'''
Author: liziwei01
Date: 2022-11-08 10:07:51
LastEditors: liziwei01
LastEditTime: 2022-12-01 21:08:59
Description: file content
'''
import glob
import os

import h5py
import numpy as np

import split_for_prepare

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

def GetFullPath(fileName):
        return os.path.join(os.getcwd(), fileName)

def GetFilePaths(dataset=TrainingDataDir):
	dataset = GetFullPath(dataset)
	data = glob.glob(os.path.join(dataset, "*.txt"))
	return data

def saveAsPreparedH5(subInputSequence, subLabelSequence, fileName="train"):
	arrData = np.asarray(subInputSequence)
	arrLabel = np.asarray(subLabelSequence)
	savePath = os.path.join(H5Dir, fileName.lower()+".h5")
	with h5py.File(savePath, 'w') as hf:
		hf.create_dataset('data', data=arrData)
		hf.create_dataset('label', data=arrLabel)

def GetH5File(fileName="train"):
	dataDir = os.path.join(H5Dir, fileName.lower()+".h5")
	with h5py.File(dataDir, "r") as hf:
		trainData = np.array(hf.get("data"))
		trainLabel = np.array(hf.get("label"))
		return trainData, trainLabel

# now input is [13, 13]
def prepareTrainingData():
	# lines of commands
	subInputSequence = []
	# label is 1 if this line of input is command injection, 0 otherwise
	subLabelSequence = []
	filePaths = GetFilePaths(TrainingDataDir)

	for filePath in filePaths:
		with open(filePath, "r") as f:
			lines = f.readlines()
			for line in lines:
				subInput = getInput(line)
				subLabel = np.array([0]).reshape([1, 1, 1])
				if filePath == isCommandInjectionFile:
					subLabel = np.array([1]).reshape([1, 1, 1])
				subInputSequence.append(subInput)
				subLabelSequence.append(subLabel)
	saveAsPreparedH5(subInputSequence, subLabelSequence, PreparedTrainingH5Name)

def getInput(line):
	res = command2Matrix(line).reshape([inputSize, inputSize, 1])
	ret = np.around(res / 128., decimals=4)
	return ret

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

# divide command into words by bash characters and ASCII control codes
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
	prepareTrainingData()
	# prepareTestingData()


def initSpecialChars():
	global SPECIAL_CHARS
	for i in range(32):
		SPECIAL_CHARS.append(chr(i))  # Add control characters
	SPECIAL_CHARS += [char for char in "~`#$&*()\\|[]}{;'\"<>/!?\n\nt"]  # Add bash special characters

if __name__ == "__main__":
	initSpecialChars()
	Prepare()

