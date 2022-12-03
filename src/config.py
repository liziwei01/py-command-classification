'''
Author: liziwei01
Date: 2022-12-03 00:46:19
LastEditors: liziwei01
LastEditTime: 2022-12-03 11:56:20
Description: file content
'''

class Config:
	ModelFileName = "model"  # Holds path from the root of the project to the model location
	TrainDir = "data/train"
	TestDir = "data/test"
	H5Dir = "data/h5"
	TrainH5FileName = "train"
	TestH5FileName = "test"
	CommandInjectionFileName = "is_command_injection"
	Epoch = 15000

config = Config()