'''
Model definition for baseline seq-to-seq model.
'''
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tf_beam_decoder import beam_decoder
import numpy as np

class ASRModel:

	def __init__(self, config):
		self.config = config
		self.build_graph()

	def build_graph(self):

		# Create placeholders
		self.add_placeholders()

		# Add embedding
		self.add_embedding()

		# Define encoder structure
		self.add_encoder()

		self.add_output_layer()

		# Define decoder structure
		self.add_decoder()

		# Branch of graph to take if we're just doing
		# prediction
		self.add_decoder_test()

		self.add_loss_op()

		self.add_training_op()

		self.add_summary_op()


	def add_placeholders(self):
		print 'Adding placeholders'
		self.input_placeholder = tf.placeholder(tf.float32, shape=(None, None, self.config.num_input_features), name='inputs')
		self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.max_out_len + 2), name="target_seq")
		self.input_seq_lens = tf.placeholder(tf.int32, shape=(None,), name='in_seq_lens')
		# self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout_keep')
		self.mask_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.max_out_len + 2), name="mask")


	def add_embedding(self):
		self.L = tf.get_variable("L", dtype=tf.float32, shape=(self.config.vocab_size, self.config.embedding_dim))

	def create_feed_dict(self, inputs, seq_lens=None, labels=None, mask=None, dropout=None):
		'''
		Creates and returns a feed dictionary since training file 
		can't easily access the model Tensors.
		'''
		feed_dict = {}

		# We always need some type of input
		feed_dict[self.input_placeholder] = inputs

		if seq_lens is not None:
			feed_dict[self.input_seq_lens] = seq_lens

		# The labels may not always be provided
		if labels is not None:
			feed_dict[self.labels_placeholder] = labels

		if mask is not None:
			feed_dict[self.mask_placeholder] = mask

		# Dropout may not actually be provided
		if dropout is not None:
			feed_dict[self.dropout_placeholder] = dropout

		return feed_dict

	def add_encoder(self):
		print 'Adding encoder'
		# Use a GRU to encode the inputs
		with tf.variable_scope('Encoder'):
			cell = tf.contrib.rnn.GRUCell(num_units = self.config.encoder_hidden_size)
			outputs, state = tf.nn.dynamic_rnn(cell, self.input_placeholder, \
											sequence_length=self.input_seq_lens, dtype=tf.float32)
			self.encoded = state

	def add_output_layer(self):
		self.output_layer = Dense(units=self.config.vocab_size, \
								kernel_initializer=tf.contrib.layers.xavier_initializer(), \
								name='fc', use_bias=True)

	def add_decoder(self):
		print 'Adding decoder'
		scope='Decoder'
		with tf.variable_scope(scope):
			cell = tf.contrib.rnn.GRUCell(num_units = self.config.decoder_hidden_size)
			# Reshape?
			decoder_inputs = tf.nn.embedding_lookup(self.L, ids=self.labels_placeholder)
			decoder_inputs = tf.unstack(decoder_inputs, axis=1)[:-1]
			outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,\
												initial_state = self.encoded,\
												cell=cell, loop_function=None, scope=scope)
			# # Get variable
			# W = tf.get_variable('W', shape=(self.config.decoder_hidden_size, self.config.vocab_size), \
			# 					initializer=tf.contrib.layers.xavier_initializer())
			# b = tf.get_variable('b', shape=(self.config.vocab_size,), \
			# 					initializer=tf.constant_initializer(0.0))
			
			# # Convert decoder inputs to a list
			# decoder_inputs = tf.unstack(self.labels_placeholder, axis=1)[:-1]
			# outputs, _ = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(decoder_inputs = decoder_inputs, \
			# 									initial_state = self.encoded, cell = cell,\
			# 									num_symbolsself.config.embedding_dim = self.config.vocab_size,\
			# 									embedding_size=self.config.embedding_dim, \
			# 									output_projection=(W, b), \
			# 									feed_previous=False)

			# Convert outputs back into Tensor
			tensor_preds = tf.stack(outputs, axis=1)
			# Compute dot product
			original_shape = tf.shape(tensor_preds)
			outputs_flat = tf.reshape(tensor_preds, [-1, self.config.decoder_hidden_size])
			# logits_flat = tf.matmul(outputs_flat, W) + b
			# logits_flat = tf.layers.dense(inputs=outputs_flat, units=self.config.vocab_size, \
								# kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc2')
			logits_flat = self.output_layer(outputs_flat)
			# Reshape back into 3D
			self.logits = tf.reshape(logits_flat, [original_shape[0], original_shape[1], self.config.vocab_size])


	'''
	Identitical to add_decoder, but geared towards decoding at test time by
	feeding in the previously generated element.
	'''
	def add_decoder_test(self):
		print 'Adding decoder test'
		scope='Decoder'
		with tf.variable_scope(scope, reuse=True):

			# Use the same cell and output projection as in the decoder train case
			cell = tf.contrib.rnn.GRUCell(num_units = self.config.decoder_hidden_size)
			# W = tf.get_variable('W')
			# b = tf.get_variable('b')

			# params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			# for param in params:
			# 	print param

			init_state = tf.tile(self.encoded, multiples=[self.config.num_beams, 1])
			print 'Init Shape', init_state
			# decoder_inputs = tf.nn.embedding_lookup(self.L, ids=self.labels_placeholder)
			beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=cell, embedding=self.L,\
													start_tokens=self.labels_placeholder[:, 0], \
													end_token=self.config.vocab_size - 1, initial_state=init_state,\
													beam_width=self.config.num_beams, output_layer=self.output_layer)

			final_outputs, final_state, final_seq_lens = tf.contrib.seq2seq.dynamic_decode(beam_search_decoder, impute_finished=False, scope=scope)
			self.test_preds = final_outputs.predicted_ids[:, :, 0]
			# # Convert input tensor to list
			# decoder_inputs = tf.unstack(self.labels_placeholder, axis=1)[:-1]

			# # Pass in to the decoder
			# outputs, _ = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(decoder_inputs = decoder_inputs, \
			# 									initial_state = self.encoded, cell = cell,\
			# 									num_symbols = self.config.vocab_size,\
			# 									embedding_size=self.config.embedding_dim, \
			# 									output_projection=(W, b), \
			# 									feed_previous=True)

			# # Convert back to tensor
			# tensor_preds = tf.stack(outputs, axis=1)

			# # Compute output_projection
			# original_shape = tf.shape(tensor_preds)
			# outputs_flat = tf.reshape(tensor_preds, [-1, self.config.decoder_hidden_size])
			# logits_flat = tf.matmul(outputs_flat, W) + b

			# # Reshape back to original
			# self.test_scores = tf.reshape(logits_flat, [original_shape[0], original_shape[1], self.config.vocab_size])
			# self.test_preds = tf.argmax(self.test_scores, axis=2)

	'''
	function: add_loss_op
	-------------------------
	Given the logits produced by the decoder, computes average loss over non-padded
	timesteps
	'''
	def add_loss_op(self):
		print 'Adding loss'
		# Compute sparse cross entropy againnst the logits
		all_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_placeholder[:, 1:])

		# First average across timestep
		masked_losses = all_losses*self.mask_placeholder[:, 1:]
		summed_losses = tf.reduce_sum(masked_losses, axis = 1)/tf.reduce_sum(self.mask_placeholder, axis = 1)

		# Then average across example
		self.loss = tf.reduce_mean(summed_losses)

		# Keep track of the change in loss
		tf.summary.scalar("Training Loss", self.loss)


	'''
	function: add_training_op
	-------------------------
	Adds the optimizer that minimizes the loss function.

	TODO: Add global norm computation
	'''
	def add_training_op(self):
		print 'Adding training op'
		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		for param in params:
			print param
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss)
		# optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
		# gvs = optimizer.compute_gradients(self.loss)
		# print gvas
  #       # gs, vs = zip(*gvs)
  #       # # Clip gradients only if self.self.config.clip_gradients is True.
  #       # if self.config.clip_gradients:
  #       #     gs, _ = tf.clip_by_global_norm(gs, self.config.max_grad_norm)
  #       #     gvs = zip(gs, vs)
  #       # # Remember to set self.grad_norm
  #       # self.grad_norm = tf.global_norm(gs)
  #       # tf.summary.scalar("Gradient Norm", self.grad_norm)
  #       self.optimizer = optimizer.apply_gradients(gvas)

    # Merges all summaries
	def add_summary_op(self):
		self.merged_summary_op = tf.summary.merge_all()

	# Trains on a single batch of input data
	def train_on_batch(self, sess, train_inputs, train_seq_len, train_targets, train_mask):
		feed_dict = self.create_feed_dict(inputs=train_inputs, seq_lens=train_seq_len, \
										labels=train_targets, mask=train_mask)
		output_dict = [self.loss, self.optimizer, self.merged_summary_op]
		loss, optimizer, summary = sess.run(output_dict, feed_dict = feed_dict)
		return loss, optimizer, summary

	# Gets loss value on a single batch of input data
	def loss_on_batch(self, sess, inputs, seq_len, targets, mask):
		feed_dict = self.create_feed_dict(inputs=inputs, seq_lens=seq_len, \
										labels=targets, mask=mask)
		output_dict = [self.loss]
		loss = sess.run(output_dict, feed_dict = feed_dict)
		return loss

	# Tests on a single batch of data
	def test_on_batch(self, sess, test_inputs, test_seq_len, test_targets):
		feed_dict = self.create_feed_dict(inputs=test_inputs, seq_lens=test_seq_len,\
										labels=test_targets)
		test_preds = sess.run(self.test_preds, feed_dict = feed_dict)
		return None, test_preds


