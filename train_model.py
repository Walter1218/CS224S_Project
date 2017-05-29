'''
This file contains the training logic for the model. It loads
a model definition from one of the directories in "models/", loads
training a validation data, and runs the training loop, saving model
weights and visualizations to the results directory.
'''
import os
import sys
import math
from time import gmtime, strftime
import time
import inspect
import tensorflow as tf
import argparse
from data_loader import DataLoader

model = None
DL_train = None
DL_val = None
'''
Function: parse_arguments()
---------------------------
Parses the command line arguments and stores/returns them in args
'''
def parse_arguments():
	parser = argparse.ArgumentParser(description='Trains an end-to-end neural ASR model')
	parser.add_argument('-m', '--model', default='baseline', help="Model you would like to train")
	parser.add_argument('-s', '--save', default=False, type=bool, help="Whether you would like to save the model, default is False.")
	parser.add_argument('-e', '--experiment', default=None, help="Name of directory you would like to save results in, default is current date")
	parser.add_argument('-rf', '--restorefile', default=None, help="What filename you would like to load the model from, default is None.")
	parser.add_argument('-r', '--restore', default=False, help="Whether you would like to restore a saved model, default is False")
	parser.add_argument('-d', '--data', default='wsj', help="What dataset you would like to use")
	parser.add_argument('-g', '--gpu', default=None, help="GPU number you would like to use")
	parser.add_argument('-n', '--normalize', default=False, type=bool, help="Whether you want to normalize MFCC features")
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
	with open(full_path + '/config.py') as f:
		print f.read()


	print 'Creating graph...'
	from model import ASRModel
	global model
	model = ASRModel()

	# Training data loader
	print 'Loading training data...'
	global DL_train
	DL_train = DataLoader(args.data, config.max_in_len, config.max_out_len, \
				normalize=args.normalize, split='train')

	# Validation data loader
	print 'Loading validation data'
	global DL_val
	DL_val = DataLoader(args.data, config.max_in_len, config.max_out_len, \
				normalize=args.normalize, mean_vector = DL_train.mean_vector ,split='dev')


def create_results_dir(args):
	parent_dir = 'results/' + args.model + '/'
	global results_dir
	if args.experiment:
		results_dir = parent_dir + args.experiment
	else:
		results_dir = parent_dir + strftime("%Y_%m_%d_%H_%M_%S", gmtime())


def train(args):

	# Init function for all variables
	init = tf.global_variables_initializer()
	config2 = tf.ConfigProto(allow_soft_placement = True)
	# Create a session
	with tf.Session(config=config2) as sess:

		# Run initialization
		sess.run(init)

		# Writes summaries out to the given path
		writer = tf.summary.FileWriter(results_dir, sess.graph)

		if args.save:
			# Saves the model to a file, or restores it
			saver = tf.train.Saver(tf.trainable_variables())

		# Load from saved model if argument is specified
		if args.restore:
			print 'Restoring'
			if args.restorefile:
				print 'Restoring from ' + args.restorefile
				saver.restore(sess, results_dir + '/' + restorefile)
			else:
				print 'No file given, restoring most recent'
				ckpt = tf.train.get_checkpoint_state(results_dir)
				if ckpt and ckpt.model_checkpoint_path:
					print 'Restoring from ' + ckpt.model_checkpoint_path
					saver.restore(sess, ckpt.model_checkpoint_path)


		# Used for keeping track of summaries
		overall_num_iters = 0

		# For every epoch
		for epoch in xrange(config.num_epochs):

			print 'Epoch ' + str(epoch) + ' of ' + str(config.num_epochs)

			# Number of batches that we loop over in one epoch
			num_iters_per_epoch = int(DL_train.num_examples/config.batch_size)

			start = time.time()
			# For every batch
			for iter_num in xrange(num_iters_per_epoch):
				# Get training batch
				batch_input, batch_seq_lens, batch_labels, batch_mask = DL_train.get_batch(batch_size=config.batch_size)
				
				# Get loss and summary
				loss, _, summary = model.train_on_batch(sess, batch_input, batch_seq_lens, batch_labels, batch_mask)
				
				# Write summary out
				writer.add_summary(summary, overall_num_iters)

				# Increment number of iterations
				overall_num_iters += 1

				# Print out training loss, iteration number, and val loss to console
				if iter_num % config.print_every == 0:
					print 'Iteration ' + str(iter_num)
					print 'Training loss is', loss
					# val_input, val_seq_lens, val_labels, val_mask = DL_val.get_batch(batch_size=config.batch_size)
#					val_loss = model.loss_on_batch(sess, val_input, val_seq_lens, val_labels, val_mask)
#					print 'Val loss is', val_loss

			print 'Epoch took ' + str(time.time() - start) + ' seconds'

			# Save after every epoch
			if args.save:
				saver.save(sess, results_dir + '/model', global_step=epoch + 1)

			print 'Sample validation results:'
			val_inputs, val_seq_lens, val_labels, val_mask = DL_val.get_batch(batch_size=5)
			val_scores, val_preds = model.test_on_batch(sess, val_inputs, val_seq_lens, val_labels)
			print 'Expected', val_labels[:, 1:22]
			print 'Got', val_preds[:, :21]


			print 'Sample train results:'
			train_inputs, train_seq_lens, train_labels, train_mask = DL_train.get_batch(batch_size=5)
			train_scores, train_preds = model.test_on_batch(sess, train_inputs, train_seq_lens, train_labels)
			print 'Expected', train_labels[:, 1:22]
			print 'Got', train_preds[:, :21]


		print 'All done!'


if __name__ == '__main__':
	args = parse_arguments()
	if args.gpu:
		with tf.device('/gpu:' + str(args.gpu)):
			print 'Attempting to run with gpu ' + str(args.gpu)
			load_model_and_data(args)
			create_results_dir(args)
			train(args)
	else:
		load_model_and_data(args)
		create_results_dir(args)
		train(args)

