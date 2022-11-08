'''
Author: liziwei01
Date: 2022-11-08 10:07:51
LastEditors: liziwei01
LastEditTime: 2022-11-08 12:53:44
Description: file content
'''
import os
import glob
import h5py
import numpy as np

TrainingDataDir = "../data/train"
TestingDataDir = "../data/test"
H5Dir = "../data/h5"
PreparedTrainingH5Name = "train"
PreparedTestingH5Name = "test"

def GetFilePaths(dataset=TrainingDataDir):
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

def prepareTrainingData():
	subInputSequence = []
	subLabelSequence = []
	filePaths = GetFilePaths(TrainingDataDir)
	# go through files
	for i in range(len(filePaths)):
		# use hash?
		pass
	saveAsPreparedH5(subInputSequence, subLabelSequence, PreparedTrainingH5Name)

def prepareTestingData():
	subInputSequence = []
	subLabelSequence = []
	filePaths = GetFilePaths(TrainingDataDir)
	# go through files
	for i in range(len(filePaths)):
		# use hash?
		pass
	saveAsPreparedH5(subInputSequence, subLabelSequence, PreparedTestingH5Name)

def Prepare():
	prepareTrainingData()
	prepareTestingData()


if __name__ == "__main__":
	Prepare()

