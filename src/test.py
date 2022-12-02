'''
Author: liziwei01
Date: 2022-12-01 23:00:24
LastEditors: liziwei01
LastEditTime: 2022-12-02 11:19:39
Description: file content
'''
'''
Author: liziwei01
Date: 2021-12-12 02:33:21
LastEditors: liziwei01
LastEditTime: 2022-05-29 03:09:18
Description: file content
'''

import time

import numpy as np
import tensorflow as tf

import prepare
import train


def getLoss(labels, pred):
	diff = np.abs(labels - pred)
	mse = np.square(diff).mean()
	return mse

'''
description: test the model and get an average false rate
return {*}
'''
def test():
	testing_data, testing_label = prepare.GetH5File(file_name=prepare.PreparedTestingH5Name)
	conv1 = train.Get2DConv(idx="1", inputs=train.Inputs, weights=train.Weights, biases=train.Biases, padding=train.Padding)
	conv2 = train.Get2DConv(idx="2", inputs=conv1, weights=train.Weights, biases=train.Biases, padding=train.Padding)
	conv3 = train.Get2DConv(idx="3", inputs=conv2, weights=train.Weights, biases=train.Biases, padding=train.Padding, activation=None)
	conv_out = conv3
	saver = tf.compat.v1.train.Saver()
	
	diff_sum = 0
	with tf.compat.v1.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir=train.CheckpointDir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("Testing...")

			startTime = time.time()
			prediction = conv_out.eval({train.Inputs: testing_data, train.Labels: testing_label})

			result = np.asarray(np.around(prediction.squeeze(), decimals=4), dtype = np.float32)
			loss = getLoss(testing_label.squeeze(), result)

			print("time: [%4.4f]" % (time.time()-startTime))
			print("avg_diff: [%.8f]" % (loss))


if __name__ == "__main__":
    test()
