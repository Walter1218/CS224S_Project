'''
Model definition for baseline seq-to-seq model.
'''
import tensorflow as tf
import numpy as np
import config

class ASRModel:

	def __init__(self):
		self.build_graph()

	def build_graph(self):

		# Create placeholders
		self.add_placeholders()

		# Define encoder structure
		self.add_encoder()

		# Define decoder structure
		self.add_decoder()

		# Branch of graph to take if we're just doing
		# prediction
		self.add_decoder_test()

		self.add_loss_op()

		self.add_training_op()

		self.add_summary_op()


	def add_placeholders(self):
		self.input_placeholder = tf.placeholder(tf.float32, shape=(None, config.max_length, config.num_input_features), name='inputs')
		self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, config.max_length), name="target_seq")
		self.input_seq_lens = tf.placeholder(tf.int32, shape=(None), name='in_seq_lens')
		self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout_keep')
		self.mask_placeholder = tf.placeholder(tf.float32, shape=(None, config.max_length), name="mask")

	def create_feed_dict(self, inputs, labels=None, seq_lens=None, mask=None dropout=None):
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

		if seq_lens is not None:
			feed_dict[self.input_seq_lens] = seq_lens

		if mask is not None:
			feed_dict[self.mask_placeholder] = mask

		# Dropout may not actually be provided
		if dropout is not None:
			feed_dict[self.dropout_placeholder] = dropout

	def add_encoder(self):
		# Use a GRU to encode the inputs
		with tf.variable_scope('Encoder'):
			cell = tf.contrib.rnn.GRUCell(num_units = config.encoder_hidden_size)
			outputs, state = tf.nn.dynamic_rnn(cell, self.input_placeholder, \
											sequence_length=self.input_seq_lens, dtype=tf.float32)
			self.encoded = state


	def add_decoder(self):
		with tf.variable_scope('Decoder'):
			cell = tf.contrib.rnn.GRUCell(num_units = config.decoder_hidden_size)

			# Get variable
			W = tf.get_variable('W', shape=(config.decoder_hidden_size, config.vocab_size), \
								initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable('b', shape=(config.vocab_size,), \
								initializer=tf.constant_initializer(0.0))

			decoder_inputs = tf.unstack(self.labels_placeholder, axis=1)
			outputs, _ = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(decoder_inputs = decoder_inputs, \
												initial_state = self.encoded, cell = cell,\
												num_symbols = config.vocab_size,\
												embedding_size=config.embedding_dim, \
												output_projection=(W, b), \
												feed_previous=False)
			tensor_preds = tf.stack(outputs, axis=1)
			original_shape = tf.shape(tensor_preds)
			outputs_flat = tf.reshape(tensor_preds, [-1, config.decoder_hidden_size])
			logits_flat = tf.matmul(outputs_flat, W) + b
			self.logits = tf.reshape(logits_flat, [original_shape[0], original_shape[1], config.vocab_size])
			print self.logits.shape

	def add_decoder_test(self):
		with tf.variable_scope('Decoder', reuse=True):
			cell = tf.contrib.rnn.GRUCell(num_units = config.decoder_hidden_size)
			W = tf.get_variable('W', shape=(config.decoder_hidden_size, config.vocab_size), initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable('b', shape=(config.vocab_size,), initializer=tf.constant_initializer(0.0))
			
			decoder_inputs = tf.unstack(self.labels_placeholder, axis=1)
			outputs, _ = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(decoder_inputs = decoder_inputs, \
												initial_state = self.encoded, cell = cell,\
												num_symbols = config.vocab_size,\
												embedding_size=config.embedding_dim, \
												output_projection=(W, b), \
												feed_previous=True)
			tensor_preds = tf.stack(outputs, axis=1)
			original_shape = tf.shape(tensor_preds)
			outputs_flat = tf.reshape(tensor_preds, [-1, config.decoder_hidden_size])
			logits_flat = tf.matmul(outputs_flat, W) + b
			self.test_preds = tf.reshape(logits_flat, [original_shape[0], original_shape[1], config.vocab_size])

	'''
	function: add_loss_op
	-------------------------
	Given the logits produced by the decoder, computes average loss over non-padded
	timesteps
	'''
	def add_loss_op(self):
		all_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_placeholder)
		masked_losses = all_losses*self.mask_placeholder
		summed_losses = tf.reduce_sum(masked_losses, axis = 1)/tf.reduce_sum(self.mask_placeholder, axis = 1)
		self.loss = tf.reduce_mean(summed_losses)
		tf.summary.scalar("Training Loss", self.loss)


	'''
	function: add_training_op
	-------------------------
	Adds the optimizer that minimizes the loss function.
	'''
	def add_training_op(self):
		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		print params
		self.optimizer = tf.train.AdamOptimizer(learning_rate=config.lr).minimize(self.loss)


	def add_summary_op(self):
		self.merged_summary_op = tf.summary.merge_all()


	def train_on_batch(self, train_inputs, train_targets, train_seq_len, train_mask):
		feed_dict = self.create_feed_dict()


