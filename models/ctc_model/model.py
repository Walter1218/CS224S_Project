'''
Model definition for baseline seq-to-seq model.
'''
import tensorflow as tf
import numpy as np
import config

tf.logging.set_verbosity(tf.logging.INFO)

class CTCModel:

	def __init__(self, args):
		# Placeholder for storing input values
		self.input_placeholder = None

		# Output ground truth sequence, of shape (batch_size, max_output_length)
		self.labels_placeholder = None

		# Dropout keep probability, scalar value
		self.dropout_placeholder = None

		# Construct the computational graph
		self.build_graph()


	def build_graph(self):

		# Create placeholders
		self.add_placeholders()

		# Add component that produces scores
		self.add_prediction_op()

		# Add the loss on top of the predictions
		self.add_loss_op()

		# Add the optimizer
		self.add_training_op()

		# Add decoding step
        self.add_decoder_and_wer_op()



	def add_placeholders(self):
		self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.num_final_features))
        self.targets_placeholder = tf.sparse_placeholder(tf.int32)
        self.seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=())


	def create_feed_dict(self, inputs, seq_lens, labels=None, dropout=None):
		'''
		Creates and returns a feed dictionary since training file 
		can't easily access the model Tensors.
		'''
		feed_dict = {}

		# We always need some type of input
		feed_dict[self.input_placeholder] = inputs

		# Sequence lengths
		feed_dict[self.seq_lens_placeholder] = seq_lens

		# The labels may not always be provided
		if labels is not None:
			feed_dict[self.labels_placeholder] = labels

		# Dropout may not actually be provided
		if dropout is not None:
			feed_dict[self.dropout_placeholder] = dropout


	def add_prediction_op(self):

		# Use a GRU cell
        cell = tf.contrib.rnn.GRUCell(config.num_hidden)

        # Run it over the inputs with a dynamic_rnn
        outputs, final_state = tf.nn.dynamic_rnn(cell, self.inputs_placeholder, dtype=tf.float32)
        
        # Initialize the projection variables
        W = tf.get_variable('W', shape=(config.num_hidden, config.num_classes), initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=(config.num_classes,), initializer=tf.constant_initializer(0.0))
        
        # Reshape the hidden states into a batch size x H*num_timesteps vector
        output_shape = tf.shape(outputs)
        flattened_outputs = tf.reshape(outputs, [-1, config.num_hidden])

        # Apply linear transformation to hidden states
        z1 = tf.matmul(flattened_outputs, W) + b

        # Reshape to get scores for every timestep
        scores = tf.reshape(z1, [output_shape[0], output_shape[1], config.num_classes])

        # Store this intermediate representation
        self.scores = scores


	def add_loss_op(self):

		# Transpose data because ctc loss expects it in a different order
		transposed_data = tf.transpose(self.logits, perm=[1, 0, 2])
        ctc_loss = tf.nn.ctc_loss(inputs=transposed_data, labels=self.targets_placeholder, sequence_length=self.seq_lens_placeholder,\
                                    preprocess_collapse_repeated=False, ctc_merge_repeated=False)
        
        # Add l2 regularization to all non-bias terms
        trainable_vars = tf.trainable_variables()
        for var in trainable_vars:
            if len(var.get_shape()) <= 1:
                continue
            l2_cost += tf.nn.l2_loss(var)

        # Remove infinity costs of training examples, no path found yet
        loss_without_invalid_paths = tf.boolean_mask(ctc_loss, tf.less(ctc_loss, tf.constant(10000.)))
        self.num_valid_examples = tf.cast(tf.shape(loss_without_invalid_paths)[0], tf.int32)
        cost = tf.reduce_mean(loss_without_invalid_paths) 

        # Store total loss
        self.loss = config.reg_val * l2_cost + cost 


    def add_decoder_and_wer_op(self):
        transposed_data = tf.transpose(self.logits, perm=[1, 0, 2])
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(transposed_data, self.seq_lens_placeholder, \
                                                        merge_repeated=False)
        decoded_sequence = tf.cast(decoded[0], tf.int32)
        all_wers = tf.edit_distance(decoded_sequence, self.targets_placeholder)
        wer = tf.reduce_mean(all_wers)

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("wer", wer)

        self.decoded_sequence = decoded_sequence
        self.wer = wer


    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()


