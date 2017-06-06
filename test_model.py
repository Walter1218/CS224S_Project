'''
This file loads a pre-trained model from the "models/" directory.
It then tests the model on the test data.
'''
import os
import sys
import math
from time import gmtime, strftime
import time
import numpy as np
import tensorflow as tf
import argparse
from data_loader import DataLoader
import editdistance

model = None
DL = None
results_dir = None

def parse_arguments():
	parser = argparse.ArgumentParser(description='Tests a pre-trained end-to-end neural ASR model')
	parser.add_argument('-m', '--model', default='baseline', help="Model you would like to evaluate")
	parser.add_argument('-e', '--expdir', default=None, help="Name of experiment you would like to evaluate")
	parser.add_argument('-rf', '--restorefile', default=None, help="What filename you would like to load the model from, default is None.")
	parser.add_argument('-d', '--data', default='wsj', help="What dataset you would like to use")
	parser.add_argument('-s', '--split', default='test', help='What split of the data you want to test on')
	parser.add_argument('-c', '--count', default=None, help='How many examples do you want to evaluate')
	parser.add_argument('-n', '--normalize', default=False, type=bool, help='Whether you want to normalize features')
	parser.add_argument('-g', '--gpu', default=None, type=int, help='Whether you want to run on a specific GPU')
	args = parser.parse_args()
	return args


def load_model_and_data(args):
	print args
	# Get model name
	model_name = args.model
	full_path = os.path.dirname(os.path.abspath(__file__))+'/models/' + model_name
	sys.path.insert(0, full_path)

	# Import the config and model from their respective files
	global config
	import config

	if args.data == 'wsj':
		config.max_in_len = 500
		config.max_out_len = 200
		config.vocab_size = 27
	elif args.data = 'chime2_grid':
		config.max_in_len = 100
		config.max_out_len = 30
		config.vocab_size = 27
	elif args.data = 'tidigits':
		config.max_in_len = 170
		config.max_out_len = 7
		config.vocab_size = 11

	print 'Current config:\n'
    variables = zip(vars(config).keys(), vars(config).values())
    for var, val in sorted(variables):
        print var + ' = ' + str(val)

	print 'Creating graph...'
	from model import ASRModel
	global model
	model = ASRModel(config)

	print 'Loading training data'
	DL_train = DataLoader(args.data, config=config, normalize=args.normalize, split='train')

	print 'Loading data'
	global DL

	# Since we've already loaded the training data for mean normalization
	if args.split == 'train':
		DL = DL_train
	else:
		DL = DataLoader(args.data, config=config, normalize=args.normalize, mean_vector=DL_train.mean_vector, split=args.split)

	global results_dir
	results_dir ='results/' + args.model + '/' + args.expdir

'''
Helper function that returns list of predictions
'''
def get_preds(sess, data, num):
	test_data = [elem[:num] for elem in data]
	i = 0
	all_preds = []
	all_labels = []
	while i < num:
		print 'i = ', i

		# Batch indices to grab
		min_i = i
		max_i = i + config.batch_size

		# Get the batch data
		input_features, seq_lens, labels, masks = tuple([elem[min_i:max_i] for elem in test_data])
		
		# Test on this batch
		scores, preds = model.test_on_batch(sess, input_features, seq_lens, labels)
		
		# Append the predictions and corresponding labels
		all_preds += list(preds)
		all_labels += list(labels)

		# Shift i
		i += config.batch_size

	# Return result
	return all_preds, all_labels


def test(args):
	# Init function for all variables
	init = tf.global_variables_initializer()

	# Allow soft placement on other GPUs
	config2 = tf.ConfigProto(allow_soft_placement = True)

	# Create a session
	with tf.Session(config=config2) as sess:

		# Run initialization
		sess.run(init)

		# Saves the model to a file, or restores it
		saver = tf.train.Saver(tf.trainable_variables())

		if args.restorefile:
			print 'Restoring from ' + args.restorefile
			saver.restore(sess, results_dir + '/' + args.restorefile)
		else:
			print 'No file given, restoring most recent'
			ckpt = tf.train.get_checkpoint_state(results_dir)
			if ckpt and ckpt.model_checkpoint_path:
				print 'Restoring from ' + ckpt.model_checkpoint_path
				saver.restore(sess, ckpt.model_checkpoint_path)

		test_data = DL.data
		total_cer = 0.0
		total_lens = 0.0
		total_wer = 0.0
		total_num_words = 0.0
		total_ser = 0.0
		num_to_evaluate = DL.num_examples

		# Use the count passed in if there is one
		if args.count is not None:
			num_to_evaluate = args.count
		print 'Evaluating ' + str(num_to_evaluate) + ' examples' 

		# Get the predictions and labels in a batch fashion
		all_preds, all_labels = get_preds(sess, test_data, int(num_to_evaluate))

		# Loop over predictions and labels and compute error rate
		for i in range(int(num_to_evaluate)):
			print 'Testing example', i
			# Input a batch of size 1
			# input_features, seq_lens, labels, mask = tuple([elem[i] for elem in test_data])
			# input_features = np.array([input_features])
			# seq_lens = np.array([seq_lens])
			# labels = np.array([labels])
			# mask = np.array([mask])

			# # Test on this "batch"
			# scores, preds = model.test_on_batch(sess, input_features, seq_lens, labels)
			# preds = preds[0]
			pred = all_preds[i]
			label = all_labels[i]
			output_pred = DL.decode(list(pred))
			output_real = DL.decode(list(label)[1:])
			print '\n'
			print 'Predicted\n', output_pred, '\n'
			print 'Real\n', output_real
			cer = editdistance.eval(output_real, output_pred)
			wer = editdistance.eval(output_real.split(), output_pred.split())
			total_cer += cer
			total_lens += len(output_real)
			total_wer += wer
			total_num_words += len(output_real.split())
			total_ser += (1 - (output_real == output_pred))

		# Print statistics
		print 'Total CER', total_cer
		print 'Total Lens', total_lens
		print 'Average CER:', total_cer/float(num_to_evaluate)
		print 'Percent CER:', total_cer/float(total_lens)
		print 'Percent WER:', total_wer/float(total_num_words)
		print 'Percent SER:', total_ser/float(num_to_evaluate)



if __name__=='__main__':
	args = parse_arguments()
	print 'Testing on ' + args.split + ' split'
	load_model_and_data(args)
	if args.gpu:
		with tf.device('/gpu:' + str(args.gpu)):
			print 'Attempting to run with gpu ' + str(args.gpu)
			test(args)
	else:		
		test(args)
