'''
This file contains the training logic for the model. It loads
a model definition from one of the directories in "models/", loads
training a validation data, and runs the training loop, saving model
weights and visualizations to the results directory.
'''
import os
import sys
import math
import tensorflow as tf
import argparse
from data_loader import DataLoader

model = None
DL = None

'''
Function: parse_arguments()
---------------------------
Parses the command line arguments and stores/returns them in args
'''
def parse_arguments():
	parser = argparse.ArgumentParser(description='Trains an end-to-end neural ASR model')
	parser.add_argument('-m', '--model', default='baseline', help="Model you would like to train")
	parser.add_argument('-s', '--save', default=None, help="Whether you would like to save the model, default is false.")
	parser.add_argument('-r', '--restore', default=None, help="What filename you would like to load the model from, default is false.")
	parser.add_argument('-d', '--data', default='./data/wsj/', help="What directory the data lives in")
	args = parser.parse_args()
	return args


'''
Function: load_model_and_data()
---------------------------
Given the command line arguments, load the appropriate model,
configuration file, and data.
'''
def load_model_and_data(args):

	# Get model name
	model_name = args.model
	full_path = os.path.dirname(os.path.abspath(__file__))+'/models/' + model_name
	sys.path.insert(0, full_path)

	# Import the config and model from their respective files
	global config
	import config


	from model import ASRModel
	global model
	model = ASRModel()

	# TODO: load train and val data
	global DL
	DL = DataLoader(path=args.data, max_in_len=config.max_in_len, max_out_len=config.max_out_len)


def train(args):

	# Init function for all variables
	init = tf.global_variables_initializer()

	# Create a session
	with tf.Session() as sess:

		# Run initialization
		sess.run(init)

		# Load from saved model if argument is specified

		# Writes summaries out to the given path
		# writer = tf.summary.FileWriter(path, sess.graph)

		# Saves the model to a file, or restores it
		saver = tf.train.Saver(tf.trainable_variables())

		# For every epoch
		for epoch in xrange(config.num_epochs):

			# Number of batches that we loop over in one epoch
			num_iters_per_epoch = int(DL.num_train_examples/config.batch_size)

			# For every batch
			for iter_num in xrange(num_iters_per_epoch):
				pass
				

				# DL.get_batch
				# 



			# Evaluate validation set, save model if better
			# saver.save(sess, args.save_to_file, global_step=curr_epoch + 1)

		# writer.add_summary(summary_extracted_from_sess, iteration number)
# 		# saver.save(session, args.save_to_file, global_step=curr_epoch + 1)
# 		# ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
# 		# if ckpt and ckpt.model_checkpoint_path:
		# saver.restore(sess, ckpt.model_checkpoint_path)
# # tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None,
# meta_graph_sufix = 'meta', write_meta_graph=True, write_state=True)



if __name__ == '__main__':
	args = parse_arguments()
	load_model_and_data(args)
	train(args)
