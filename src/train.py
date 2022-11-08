'''
Author: liziwei01
Date: 2022-11-08 12:31:40
LastEditors: liziwei01
LastEditTime: 2022-11-08 13:10:51
Description: file content
'''
import prepare
import tensorflow as tf
import time

### configuration
epoch = 15000
padding = "VALID"
checkpointDir = "../data/checkpoint"
###

ReLU = "ReLU"
normalStrides = [1,1,1,1]
placeholderShape = [None]
inputShape = [None]
stddev = 1e-3
trainable = True

tf.compat.v1.disable_eager_execution()
Inputs = tf.compat.v1.placeholder(tf.float32, placeholderShape, name="inputs")
Labels = tf.compat.v1.placeholder(tf.float32, placeholderShape, name="labels")
weights = {
	"w1": tf.Variable(initial_value=tf.random.normal(inputShape, stddev=stddev), trainable=trainable, name="w1")
}
biases = {
	"b1": tf.Variable(initial_value=tf.zeros([inputShape[len(inputShape)-1]]),trainable=trainable ,name="b1")
}
optimizer = {
	"o1": tf.compat.v1.train.GradientDescentOptimizer(stddev)
}

def getLoss(labels, pred):
	# mse
	return tf.reduce_mean(input_tensor=tf.square(labels - pred))

def get2DConv(idx, inputs, weights, biases, padding, strides=normalStrides, activation=ReLU):
	conv = tf.nn.conv2d(input=inputs, filters=weights["w"+idx], strides=strides, padding=padding) + biases["b"+idx]
	if activation == ReLU:
		conv = tf.nn.relu(conv)
	return conv

def train():
	trainingData, trainingLabel = prepare.GetH5File(fileName=prepare.PreparedTrainingH5Name)
	conv1 = get2DConv(idx="1", inputs=Inputs, weights=weights, biases=biases, padding=padding)
	# conv2...
	conv_out = conv1

	var_list1 = [weights["w1"], biases["b1"]]
	# var_list2...
	
	loss = getLoss(Labels, conv_out)
	grads = tf.gradients(ys=loss, xs=var_list1)
	grads1 = grads
	# grads2...

	train_op1 = optimizer["o1"].apply_gradients(zip(grads1, var_list1))
	train_op = tf.group(train_op1) # train_op2...

	counter = 0
	start_time = time.time()
	saver=tf.compat.v1.train.Saver(max_to_keep=5)

	with tf.compat.v1.Session() as sess:
		print("Training...")
		sess.run(tf.compat.v1.initialize_all_variables())
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpointDir)
		if ckpt and ckpt.model_checkpoint_path:
			print("Continuing")
			saver.restore(sess, ckpt.model_checkpoint_path)
		
		for ep in range(epoch):
			pass

if __name__ == "__main__":
	train()
