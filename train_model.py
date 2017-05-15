'''
This file contains the training logic for the model. It loads
a model definition from one of the directories in "models/", loads
training a validation data, and runs the training loop, saving model
weights and visualizations to the results directory.
'''

import tensorflow as tf
import argparse


def parse_arguments():
	parser = argparse.ArgumentParser(description='Trains an end-to-end neural ASR model')
	parser.add_argument('-m', '--model', help="Model you would like to train")
	args = parser.parse_args()


def train(model):


# 	init = tf.global_variables_initializer()
# 	with tf.Session as sess:
# 		sess.run(init)
#		writer = tf.summary.FileWriter(path, sess.graph)
# 		# writer.add_summary(summary_extracted_from_sess, iteration number)
# 		# saver = tf.train.Saver(tf.trainable_variables())
# 		# saver.save(session, args.save_to_file, global_step=curr_epoch + 1)
# 		# ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
# 		# if ckpt and ckpt.model_checkpoint_path:
		# saver.restore(sess, ckpt.model_checkpoint_path)
# # tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None,
# meta_graph_sufix = 'meta', write_meta_graph=True, write_state=True)
	pass



if __name__ == '__main__':

	parse_arguments()
