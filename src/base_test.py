'''
Author: bfishe32
Date: 2022-12-03 00:46:19
LastEditors: liziwei01
LastEditTime: 2022-12-03 11:42:03
Description: file content
'''

"""Module that allows interfacing with the model"""
import numpy as np
from keras.models import load_model
from os import path
from config import config

model = None


def load():
	"""Loads Model for Use."""
	global model
	if model is None:
		model = load_model(path.join(config.H5Dir, config.ModelFileName+".h5"))

def predict_is_ci_str(string : str):
	"""Predict if Command Injection

	Args:
		string (str): String to predict if it's command injection. Is automatically shortened/lengthened to 100 characters.

	Returns:
		tf.Tensor: A tensor containing a bool of True or False. Can be used in == compares, but not `is` compares.
	"""
	string = string[0:100].strip()
	while len(string) < 100:
		string += " "
	command = np.array([ord(c) for c in string])

	return predict_is_ci(command)

def predict_is_ci(command):
	"""Predict if Command Injection

	Args:
		command (np.ndarray): Command to predict if it's command injection.

	Returns:
		tf.Tensor: A tensor containing a bool of True or False. Can be used in == compares, but not `is` compares.
	"""
	return get_ci_certainty(command.reshape(1, len(command))) >= 0.5

def get_ci_certainty(command):
	"""Get Command Injection Certainty.

	Args:
		string (str): String to get certainty of command injection. Is automatically shortened/lengthened to 100 characters.

	Returns:
		float: Effectively the percentage chance that string is command injection
	"""
	global model
	if model is None:
		load()
	predicted = model(command)
	predicted = predicted[0][0]
	return predicted