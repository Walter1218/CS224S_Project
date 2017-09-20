'''
Model definition for baseline seq-to-seq model.
'''
import tensorflow as tf
from tf_beam_decoder import beam_decoder
import numpy as np
from my_att_cell import MyAttCell

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

		self.add_cell()

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
		self.input_seq_lens = tf.placeholder(tf.int32, shape=(None), name='in_seq_lens')
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


	def concatenate(self, inputs, sequence_lengths):
		input_shape = tf.Tensor.get_shape(inputs)
		print('concat: initial shape: ', input_shape)
		concat_inputs = []
		for time_i in range(1, int(input_shape[1]), 2):
		    concat_input = tf.concat(1, [inputs[:, time_i-1, :], inputs[:, time_i, :]], name='plstm_concat')
		    concat_inputs.append(concat_input)
		inputs = tf.pack(concat_inputs, axis=1, name='plstm_pack')
		concat_shape = tf.Tensor.get_shape(inputs)
		print('concat: concat shape: ', concat_shape)

		sequence_lengths = tf.cast(tf.floor(tf.cast(sequence_lengths, tf.float32)/2), tf.int32)
		return inputs, sequence_lengths


	def add_encoder(self):
		print 'Adding encoder'
		# Use a GRU to encode the inputs
		with tf.variable_scope('Encoder'):

			# Base BLSTM
			cell_fw = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)
			cell_bw = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)
			# in_lens = tf.cast(tf.ceil(tf.cast(self.input_seq_lens, tf.float32)/2.0), tf.int32)
			outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw, cell_bw = cell_bw, sequence_length=self.input_seq_lens, inputs=self.input_placeholder, dtype=tf.float32)
			h1_vals = tf.concat(2, outputs)

			# First PLSTM
			concat_inputs, new_seq_lens = self.concatenate(h1_vals, self.input_seq_lens)

			cell_fw2 = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)
			cell_bw2 = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)

			outputs2, states2 = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw2, cell_bw = cell_bw2, sequence_length=new_seq_lens, inputs=concat_inputs, dtype=tf.float32)
			
			# Pass concatenated hidden states as inputs to CNN
			h2_vals = tf.concat(2, outputs2)


			# Second PLSTM
			concat_inputs2, new_seq_lens2 = self.concatenate(h2_vals, new_seq_lens)

			cell_fw3 = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)
			cell_bw3 = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)

			outputs3, states3 = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw3, cell_bw = cell_bw3, sequence_length=new_seq_lens2, inputs=concat_inputs2, dtype=tf.float32)
			
			# Pass concatenated hidden states as inputs to CNN
			h3_vals = tf.concat(2, outputs3)

			# Final PLSTM
			concat_inputs3, new_seq_lens3 = self.concatenate(h3_vals, new_seq_lens2)
			
			# Lastly, run a single dynamic rnn over the resulting time series, and use
			# corresponding states as memory
			forward_cells = []
			for i in range(self.config.num_layers):
				cell = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)
				forward_cells.append(cell)
			cell_fw3 = tf.nn.rnn_cell.MultiRNNCell(cells=forward_cells)
			backward_cells = []
			for i in range(self.config.num_layers):
				cell = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)
				backward_cells.append(cell)
			cell_bw3 = tf.nn.rnn_cell.MultiRNNCell(cells=backward_cells)
			final_outputs, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell_fw3, cell_bw = cell_bw3, sequence_length=new_seq_lens3, inputs=concat_inputs3, \
											dtype=tf.float32)
			all_states = []
			for i in range(self.config.num_layers):
				all_states.append(tf.concat(1, (final_states[0][i], final_states[1][i])))
			print 'All states', all_states
			self.encoded = tuple(all_states)
			print self.encoded
			self.memory = tf.concat(2, final_outputs)
			print 'Memory shape', self.memory.get_shape()
			# cell_fw2 = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)
			# cell_bw2 = tf.nn.rnn_cell.GRUCell(num_units = self.config.encoder_hidden_size)
			# final_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw2, cell_bw=cell_bw2, inputs=conv2, dtype=tf.float32)
			# self.encoded = final_state
			# self.memory = final_outputs
			# print 'Memory shape', self.memory.get_shape()
			# print 'Encoded shape', self.encoded.get_shape()


	def add_cell(self):
		cells = []
		for i in range(self.config.num_layers):
			cell = tf.nn.rnn_cell.GRUCell(num_units=self.config.decoder_hidden_size)
			cells.append(cell)
		cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
		self.cell = MyAttCell(memory=self.memory, num_units=self.config.decoder_hidden_size, cell=cell, config=self.config)
		print 'Cell state size', self.cell.state_size


	def add_decoder(self):
		print 'Adding decoder'
		scope='Decoder'
		with tf.variable_scope(scope):

			W = tf.get_variable('W', shape=(self.config.decoder_hidden_size, self.config.vocab_size), \
								initializer=tf.contrib.layers.xavier_initializer())
			b = tf.get_variable('b', shape=(self.config.vocab_size,), \
								initializer=tf.constant_initializer(0.0))

			# Greedy decoder
			def loop_fn(prev, i):
				indices = tf.argmax(tf.matmul(prev, W) + b, axis=1)
				return tf.nn.embedding_lookup(self.L, indices)

			loop = None
			if self.config.loop:
				loop = loop_fn

			# Reshape
			decoder_inputs = tf.nn.embedding_lookup(self.L, ids=self.labels_placeholder)
			decoder_inputs = tf.unstack(decoder_inputs, axis=1)[:-1]
			init_state = list(self.encoded) + [tf.zeros_like(self.encoded[0])]
			# for i in range(self.config.num_dec_layers):
			# 	init_state.append(tf.zeros_like(self.encoded, dtype=tf.float32))
			init_state = tuple(init_state)
			outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,\
												initial_state = init_state,\
												cell=self.cell, loop_function=loop, scope=scope)		

			# Convert outputs back into Tensor
			tensor_preds = tf.stack(outputs, axis=1)
			tensor_preds = tf.nn.dropout(tensor_preds, keep_prob=self.config.dropout_p)
			
			# Compute dot product
			original_shape = tf.shape(tensor_preds)
			outputs_flat = tf.reshape(tensor_preds, [-1, self.config.decoder_hidden_size])
			logits_flat = tf.matmul(outputs_flat, W) + b

			# Reshape back into 3D
			self.logits = tf.reshape(logits_flat, [original_shape[0], original_shape[1], self.config.vocab_size])
			print 'Logits shape', self.logits.get_shape()


	'''
	Identitical to add_decoder, but geared towards decoding at test time by
	feeding in the previously generated element.
	'''
	def add_decoder_test(self):
		print 'Adding decoder test'
		scope='Decoder'
		with tf.variable_scope(scope, reuse=True):

			# Use the same output projection as in the decoder train case
			W = tf.get_variable('W')
			b = tf.get_variable('b')

			def output_fn(inputs):
				original_shape = tf.shape(inputs)
				outputs_flat = tf.reshape(inputs, [-1, self.config.decoder_hidden_size])
				logits_flat = tf.matmul(outputs_flat, W) + b
				logits = tf.reshape(logits_flat, [original_shape[0], original_shape[1], self.config.vocab_size])
				return tf.nn.log_softmax(logits)

			def emb_fn(tokens):
				original_shape = tf.shape(tokens)
				outputs = tf.nn.embedding_lookup(self.L, tokens)
				return tf.reshape(outputs, [original_shape[0], original_shape[1], self.config.embedding_dim])

			start_tokens = tf.nn.embedding_lookup(self.L, self.labels_placeholder[:, 0])
			init_state = list(self.encoded) + [tf.zeros_like(self.encoded[0])]
			# init_state = [self.encoded]
			# for i in range(self.config.num_dec_layers):
			# 	init_state.append(tf.zeros_like(self.encoded, dtype=tf.float32))
			init_state = tuple(init_state)
			self.decoded, _ = beam_decoder(
			    cell=self.cell,
			    beam_size=self.config.num_beams,
			    stop_token=self.config.vocab_size - 1,
			    initial_state=init_state,
			    initial_input=start_tokens,
			    tokens_to_inputs_fn=emb_fn,
			    max_len=self.config.max_out_len,
			    scope=scope,
			    outputs_to_score_fn=output_fn,
			    output_dense=True,
			    cell_transform='replicate',
			    score_upper_bound = self.config.beam_threshold
			)


			# Greedy decoder
			def loop_fn(prev, i):
				indices = tf.argmax(tf.matmul(prev, W) + b, axis=1)
				return tf.nn.embedding_lookup(self.L, indices)


			decoder_inputs = tf.nn.embedding_lookup(self.L, ids=self.labels_placeholder)
			decoder_inputs = tf.unstack(decoder_inputs, axis=1)[:-1]
			outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,\
												initial_state = init_state,\
												cell=self.cell, loop_function=loop_fn, scope=scope)

			# Convert back to tensor
			tensor_preds = tf.stack(outputs, axis=1)

			# Compute output_projection
			original_shape = tf.shape(tensor_preds)
			outputs_flat = tf.reshape(tensor_preds, [-1, self.config.decoder_hidden_size])
			logits_flat = tf.matmul(outputs_flat, W) + b

			# Reshape back to original
			self.test_scores = tf.reshape(logits_flat, [original_shape[0], original_shape[1], self.config.vocab_size])
			self.greedy_decoded = tf.argmax(self.test_scores, axis=2)



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

		global_step = tf.Variable(0, trainable=False)
		self.lr = tf.train.exponential_decay(self.config.lr, global_step,
                                             5000, 0.70, staircase=True)
		tf.summary.scalar("Learning Rate", self.lr)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=global_step)

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
		test_preds = sess.run(self.greedy_decoded, feed_dict = feed_dict)
		return None, test_preds

	# Tests on a single batch of data
	def test_beam_on_batch(self, sess, test_inputs, test_seq_len, test_targets):
		feed_dict = self.create_feed_dict(inputs=test_inputs, seq_lens=test_seq_len,\
										labels=test_targets)
		test_preds = sess.run(self.decoded, feed_dict = feed_dict)
		return None, test_preds





