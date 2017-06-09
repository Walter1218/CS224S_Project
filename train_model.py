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
	parser.add_argument('-b', '--batch_size', default=None, type=int, help="How many examples per batch")
	parser.add_argument('-emb', '--embedding_size', default=None, type=int, help="How large the embedding dimension should be")
	parser.add_argument('-l', '--loop', default=None, help="How large the embedding dimension should be")
	parser.add_argument('-nl', '--num_layers', default=1, type=int, help="How many layers to use for encoder and decoder")
	parser.add_argument('-nc', '--num_cells', default=64, type=int, help="How many cells to use for the memory-based models.")
	parser.add_argument('-bt', '--beam_threshold', default=0.0, type=float, help="What threshold to use during beamsearch")
	parser.add_argument('-dp', '--dropout_p', default=None, type=float, help="What keep probability to use for dropout")
	parser.add_argument('-cg', '--clip_gradients', default=None, help="Whether to clip gradients by global norm")	
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
	config.loop = args.loop
	config.num_layers = args.num_layers
	config.num_cells = args.num_cells
	config.beam_threshold = float(args.beam_threshold)
	if args.clip_gradients is not None:
		config.clip_gradients = bool(args.clip_gradients)
	# config.num_dec_layers = args.num_dec_layers
	if args.data == 'wsj':
		config.max_in_len = 500
		config.max_out_len = 200
		config.vocab_size = 27
	elif args.data == 'chime2_grid':
		config.max_in_len = 100
		config.max_out_len = 30
		config.vocab_size = 27
	elif args.data == 'tidigits':
		config.max_in_len = 170
		config.max_out_len = 7
		config.vocab_size = 11

	config.vocab_size = config.vocab_size + 3 #Special token for start, end, and pad
	if args.batch_size:
		config.batch_size = int(args.batch_size)

	if args.embedding_size:
		config.embedding_dim = int(args.embedding_size)
	if args.dropout_p:
		config.dropout_p = float(args.dropout_p)
	print 'Current config:\n'
	variables = zip(vars(config).keys(), vars(config).values())
	for var, val in sorted(variables):
		print var + ' = ' + str(val)


	print 'Creating graph...'
	from model import ASRModel
	global model
	model = ASRModel(config)

	# Training data loader
	print 'Loading training data...'
	global DL_train
	DL_train = DataLoader(args.data, config=config, \
				normalize=args.normalize, split='train')

	# Validation data loader
	print 'Loading validation data'
	global DL_val
	DL_val = DataLoader(args.data, config=config, \
				normalize=args.normalize, mean_vector = DL_train.mean_vector, split='dev')


def create_results_dir(args):
	parent_dir = 'results/' + args.model + '/'
	global results_dir
	if args.experiment:
		results_dir = parent_dir + args.experiment
	else:
		results_dir = parent_dir + strftime("%Y_%m_%d_%H_%M_%S", gmtime())


def print_examples(labels, preds, DL):
	for i in range(5):
		print 'Expected'
		print DL.decode(list(labels[i])[1:])
		print 'Got'
		print DL.decode(list(preds[i])) + '\n'

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
			print_examples(val_labels, val_preds, DL_val)



			print 'Sample train results:'
			train_inputs, train_seq_lens, train_labels, train_mask = DL_train.get_batch(batch_size=5)
			train_scores, train_preds = model.test_on_batch(sess, train_inputs, train_seq_lens, train_labels)
			print_examples(train_labels, train_preds, DL_train)

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

