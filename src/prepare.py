'''
Author: liziwei01
Date: 2022-11-08 10:07:51
LastEditors: liziwei01
LastEditTime: 2022-11-27 07:51:03
Description: file content
'''
import os
import glob
import h5py
import numpy as np

import split_for_prepare

TrainingDataDir = "data/train/"
TestingDataDir = "data/test/"
isCommandInjectionFile = "is_command_injection.txt"
notCommandInjectionFile = ""
H5Dir = "data/h5"
PreparedTrainingH5Name = "train"
PreparedTestingH5Name = "test"

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
				subInput = command2Matrix(line)
				subLabel = [[[1]]]
				subInputSequence.append(subInput)
				subLabelSequence.append(subLabel)
	saveAsPreparedH5(subInputSequence, subLabelSequence, PreparedTrainingH5Name)

def command2Matrix(command):
	# divide line to words
	words = line2words(command)
	# add placeholder
	for i in range(inputSize):
		words.append([0])
	for i in range(inputSize):
		for j in range(inputSize):
			words[i].append(0)
		words[i] = words[i][:inputSize]
	
	return [words[:inputSize]]

# divide command into words by
# ' ', ';', '&lt;', '|', '||' '&', '&&', '\n' etc
def line2words(command):
	line = []
	lines = []
	for i in range(len(command)):
		if i == len(command)-1:
			lines.append(line)
		if command[i] == " ":
			lines.append(line)
			line = []
		elif command[i] == ";":
			lines.append(line)
			line = []
		else:
			char = ord(command[i])
			line.append(char)
	return lines


def Prepare():
	prepareTrainingData()
	# prepareTestingData()


if __name__ == "__main__":
	Prepare()

