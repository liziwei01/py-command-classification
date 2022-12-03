'''
Author: liziwei01
Date: 2022-12-03 00:46:19
LastEditors: liziwei01
LastEditTime: 2022-12-03 11:55:02
Description: file content
'''
"""Tests model on testing data."""
from config import config
import prepare
import base_test

def test():
	testing_data, testing_labels = prepare.GetH5File(config.TestH5FileName)
	correct = 0
	incorrect = 0
	correct_positive = 0
	incorrect_positive = 0
	correct_negative = 0
	incorrect_negative = 0

	for i in range(len(testing_data)):
		print("Testing {} of {}".format(i + 1, len(testing_data)), end="\r")
		predicted = base_test.predict_is_ci(testing_data[i])
		if predicted == (testing_labels[i][0] == 1):
			correct += 1
			if testing_labels[i][0] == 1:
				correct_positive += 1
			else:
				correct_negative += 1
		else:
			incorrect += 1
			if testing_labels[i][0] == 1:
				incorrect_positive += 1
			else:
				incorrect_negative += 1

	print("\nTotal Correct: {}/{} ({}%)".format(correct, correct + incorrect, correct/(correct + incorrect) * 100))
	print("Correct CI Identifications: {}/{} ({}%)".format(correct_positive, correct_positive + incorrect_positive, correct_positive / (correct_positive + incorrect_positive) * 100))
	print("Correct non-CI Identifications: {}/{} ({}%)".format(correct_negative, correct_negative + incorrect_negative, correct_negative / (correct_negative + incorrect_negative) * 100))

if __name__ == "__main__":
	test()