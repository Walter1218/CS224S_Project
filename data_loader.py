import pickle
import numpy as np
# Load minibatch function

#input
#labels
#mask
#seq

class DataLoader:

	def __init__(self, dataset, config=None, normalize=False, mean_vector=None, split='train'):
		# Load the data from pickle file depending on input
		self.max_input = config.max_in_len
		self.max_output = config.max_out_len
		self.normalize = normalize
		self.mean_vector = mean_vector
		self.split = split
		self.pad_char = config.vocab_size - 3 #27
		self.start_char = config.vocab_size - 2# 28
		self.end_char = config.vocab_size - 1#29

		self.ints_to_chars = {}
		self.create_mapping(dataset)

		# Load the data depending on what split was specified
		pathname = 'data/' + dataset+'/'
		if self.split == 'train':
			features = pickle.load(open(pathname+'mfcc_train.pkl', 'rb'))
			labels = pickle.load(open(pathname+'labels_train.pkl', 'rb'))
		elif self.split == 'dev':
			features = pickle.load(open(pathname+'mfcc_dev.pkl', 'rb'))
			labels = pickle.load(open(pathname+'labels_dev.pkl', 'rb'))
		else:
			features = pickle.load(open(pathname+'mfcc_test.pkl', 'rb'))
			labels = pickle.load(open(pathname+'labels_test.pkl', 'rb'))

		# Lists for loading the data one datapoint at a time
		self.input_ids = []
		self.batch_features = []
		self.batch_labels = []
		self.sequence_lens = []
		self.masks = []
		self.mean_vector = mean_vector

		# We will use this to compute the average feature
		feature_sum = np.zeros(39)
		feature_count = 0.0
		keys = sorted(features.keys())
		for f in keys:

			# Get the feature using the id as a key
			# This has shape[num_features, number of timesteps]
			feature = features[f]

			# Ignore examples that fall out of the specified range
			if feature.shape[1] > self.max_input or len(labels[f]) > self.max_output: 
				continue

			# Add to the growing sum of feature vectors
			# by adding in the sum of the sequence feature vectors
			feature_sum += np.sum(feature, axis=1)

			# Append padding to input features, and get sequence length
			feature, sequence_len = self.pad_feature(feature)

			# The feature count is the number of sequences seen so far
			feature_count += sequence_len

			# Append padding to the output
			label, mask = self.pad_label(labels[f])

			# Store results
			self.input_ids.append(f)
			self.batch_features.append(feature)
			self.batch_labels.append(label)
			self.sequence_lens.append(sequence_len)
			self.masks.append(mask)

		# Convert lists to numpy arrays

		# Shape: number of examples
		self.input_ids = np.array(self.input_ids)

		# Shape: [Num Examples, max_output_length]
		self.batch_labels = np.array(self.batch_labels)

		# Shape: [Num Examples, Num Timesteps, Num Features]
		self.batch_features = np.array(self.batch_features)

		# Normalize features if you want
		if self.normalize:
			print 'Normalizing'
			# If mean vector is not given
			if self.mean_vector is None:
				print 'Computing Mean Vector'
				# self.mean_vector = np.mean(np.mean(self.batch_features, axis=2), axis=0)
				# Obtain the mean vector by averaging
				self.mean_vector = feature_sum/float(feature_count)
			# self.batch_features = (self.batch_features.transpose((0,2,1)) - self.mean_vector).transpose((0,2,1))
			self.batch_features = self.batch_features - self.mean_vector

		self.num_examples = self.batch_features.shape[0]
		
		self.sequence_lens = np.array(self.sequence_lens)
		self.masks = np.array(self.masks)

		self.data = [self.batch_features, self.sequence_lens, self.batch_labels, self.masks]
		
		print 'Loaded ' + str(self.num_examples) + ' examples!'


	def pad_feature(self, feature, side='post'):

		# Amount of padding needed
		pad = np.zeros((feature.shape[0], self.max_input - feature.shape[1]))

		# Do padding at the beginning
		if side == 'pre':
			return np.transpose(np.append(pad, feature, 1)), feature.shape[1]

		# Do padding at the end
		else:
			return np.transpose(np.append(feature, pad, 1)), feature.shape[1]


	def pad_label(self, label, side='post'):
		# The amount of padding needed
		pad = [self.pad_char]*(self.max_output - len(label))

		# Do padding at the beginning
		if side == 'pre':
			mask = [0] * len(pad) + [1] * (len(label)+2)
			return pad + [self.start_char] + label + [self.end_char], mask

		# Do padding at the end
		else:
			mask = [1] * (len(label)+2)+[0] * len(pad)
			return [self.start_char] + label + [self.end_char] + pad, mask

	'''
	Returns a random batch of data
	'''
	def get_batch(self, batch_size=32, shuffle=True):
	    rand_indices = np.random.choice(range(self.num_examples), size=batch_size, replace=False)
	    return self.batch_features[rand_indices], self.sequence_lens[rand_indices], self.batch_labels[rand_indices], \
	    		self.masks[rand_indices]

	def create_mapping(self, dataset):
		if dataset == 'tidigits':
			for i in range(10):
				self.ints_to_chars[i] = str(i)
			self.ints_to_chars[10] = '0'
		else:
			for i in range(26):
				self.ints_to_chars[i] = chr(i + 97)
			self.ints_to_chars[26] = ' '
		self.ints_to_chars[self.pad_char] = '<PAD>'
		self.ints_to_chars[self.start_char] = '<s>'
		self.ints_to_chars[self.end_char] = '<e>'

	'''
	Given an input sequences of indices, converts them into characters, and stops when
	reaches a stop token (includes the stop token in the output)
	'''
	def decode(self, input_seq):
		output = ''
		for val in input_seq:
			next_char = self.ints_to_chars[val]
			output += self.ints_to_chars[val]
			if val == self.end_char:
				break
		return output


