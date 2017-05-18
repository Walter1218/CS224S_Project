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
DL_test = None
results_dir = None

def parse_arguments():
	parser = argparse.ArgumentParser(description='Tests a pre-trained end-to-end neural ASR model')
	parser.add_argument('-m', '--model', default='baseline', help="Model you would like to evaluate")
	parser.add_argument('-e', '--expdir', default=None, help="Name of experiment you would like to evaluate")
	parser.add_argument('-rf', '--restorefile', default=None, help="What filename you would like to load the model from, default is None.")
	parser.add_argument('-d', '--data', default='wsj', help="What dataset you would like to use")
	args = parser.parse_args()
	return args


def load_model_and_data(args):
	# Get model name
	model_name = args.model
	full_path = os.path.dirname(os.path.abspath(__file__))+'/models/' + model_name
	sys.path.insert(0, full_path)

	# Import the config and model from their respective files
	global config
	import config

	print 'Creating graph...'
	from model import ASRModel
	global model
	model = ASRModel()

	print 'Loading test data'
	global DL_test
	DL_test = DataLoader(args.data, config.max_in_len, config.max_out_len, normalize=False, split='test')


	global results_dir
	results_dir ='results/' + args.model + '/' + args.expdir


def test(args):
	# Init function for all variables
	init = tf.global_variables_initializer()

	# Create a session
	with tf.Session() as sess:

		# Run initialization
		sess.run(init)

		# Saves the model to a file, or restores it
		saver = tf.train.Saver(tf.trainable_variables())

		if args.restorefile:
			print 'Restoring from ' + args.restorefile
			saver.restore(sess, results_dir + '/' + restorefile)
		else:
			print 'No file given, restoring most recent'
			ckpt = tf.train.get_checkpoint_state(results_dir)
			if ckpt and ckpt.model_checkpoint_path:
				print 'Restoring from ' + ckpt.model_checkpoint_path
				saver.restore(sess, ckpt.model_checkpoint_path)

		test_data = DL_test.data
		total_cer = 0.0
		for i in range(DL_test.num_examples):
			print 'Testing example', i
			# Input a batch of size 1
			input_features, seq_lens, labels, mask = tuple([elem[i] for elem in test_data])
			input_features = np.array([input_features])
			seq_lens = np.array([seq_lens])
			labels = np.array([labels])
			mask = np.array([mask])

			# Test on this "batch"
			scores, preds = model.test_on_batch(sess, input_features, seq_lens, labels)
			scores = scores[0]
			preds = preds[0]
			output_pred = DL_test.decode(preds)
			output_real = DL_test.decode(list(labels[0]))
                        print 'Predicted\n', output_pred
                        print 'Real\n', output_real
			cer = editdistance.eval(output_real, output_pred)
			total_cer += cer
		print 'Average CER:', total_cer/num_examples





if __name__=='__main__':
	args = parse_arguments()
	load_model_and_data(args)
	test(args)
