'''
Model definition for baseline seq-to-seq model.
'''
import tensorflow as tf
import numpy as np
import config

class Baseline:

	def __init__(self, args):
		# Placeholder for storing input values
		self.input_placeholder = None

		# Output ground truth sequence, of shape (batch_size, max_output_length)
		self.labels_placeholder = None

		# Dropout keep probability, scalar value
		self.dropout_placeholder = None


		self.build_graph()



	def build_graph(self):

		# Create placeholders
		self.add_placeholders()

		# Define encoder structure
		self.add_encoder()

		# Define decoder structure
		self.add_decoder()


	def add_placeholders(self):
		pass


	def create_feed_dict(self, inputs, labels=None, dropout=None):
		'''
		Creates and returns a feed dictionary since training file 
		can't easily access the model Tensors.
		'''
		feed_dict = {}

		# We always need some type of input
		feed_dict[self.input_placeholder] = inputs

		# The labels may not always be provided
		if labels is not None:
			feed_dict[self.labels_placeholder] = labels

		# Dropout may not actually be provided
		if dropout is not None:
			feed_dict[self.dropout_placeholder] = dropout


	def add_encoder(self):

		# Build a GRU on top of the 
		cell = tf.contrib.rnn.GRUCell(num_units = config.encoder_hidden_size)
		outputs_state = tf.nn.dynamic_rnn()


	def add_decoder(self):
		pass


	def add_prediction_op(self):
		pass


	def add_loss_op(self):
		pass


