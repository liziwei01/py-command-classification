'''
Author: liziwei01
Date: 2022-11-08 10:07:51
LastEditors: liziwei01
LastEditTime: 2022-11-08 10:40:45
Description: file content
'''
import os
import glob
import h5py
import numpy as np

class Data:
    TrainingDataDir = "../../data/train"
    TestingDataDir = "../../data/test"
    H5Dir = "../../data/h5"
    PreparedTrainingH5Name = "train"
    PreparedTestingH5Name = "test"

    def __init__(self):
        pass

    def GetFilePaths(self, dataset=TrainingDataDir):
        data = glob.glob(os.path.join(dataset, "*.txt"))
        return data

    def __saveAsPreparedH5(self, subInputSequence, subLabelSequence, fileName="train"):
        arrData = np.asarray(subInputSequence)
        arrLabel = np.asarray(subLabelSequence)
        savePath = os.path.join(self.H5Dir, fileName.lower()+".h5")
        with h5py.File(savePath, 'w') as hf:
            hf.create_dataset('data', data=arrData)
            hf.create_dataset('label', data=arrLabel)

    def GetH5File(self, fileName="train"):
        dataDir = os.path.join(self.H5Dir, fileName.lower()+".h5")
        with h5py.File(dataDir, "r") as hf:
            trainData = np.array(hf.get("data"))
            trainLabel = np.array(hf.get("label"))
            return trainData, trainLabel

    def PrepareTrainingData(self):
        subInputSequence = []
        subLabelSequence = []
        filePaths = self.GetFilePaths(self.TrainingDataDir)
        # go through files
        for i in range(len(filePaths)):
            # use hash?
            pass
        self.__saveAsPreparedH5(subInputSequence, subLabelSequence, self.PreparedTrainingH5Name)

    def PrepareTestingData(self):
        subInputSequence = []
        subLabelSequence = []
        filePaths = self.GetFilePaths(self.TrainingDataDir)
        # go through files
        for i in range(len(filePaths)):
            # use hash?
            pass
        self.__saveAsPreparedH5(subInputSequence, subLabelSequence, self.PreparedTestingH5Name)


if __name__ == "__main__":
    D = Data()
    D.PrepareTrainingData()
    D.PrepareTestingData()

