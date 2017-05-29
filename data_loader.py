import pickle
import numpy as np
# Load minibatch function

#input
#labels
#mask
#seq

class DataLoader:

	def __init__(self, dataset, max_input, max_output, normalize=False, mean_vector=None, split='train'):
		# Load the data from pickle file depending on input
		self.max_input = max_input
		self.max_output = max_output
		self.normalize = normalize
		self.split = split
		self.pad_char = 27
		self.start_char = 28
		self.end_char = 29

		self.ints_to_chars = {}
		for i in range(26):
			self.ints_to_chars[i] = chr(i + 97)
		self.ints_to_chars[26] = ' '
		self.ints_to_chars[self.pad_char] = '<PAD>'
		self.ints_to_chars[self.start_char] = '<s>'
		self.ints_to_chars[self.end_char] = '<e>'

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

		self.input_ids = []
		self.batch_features = []
		self.batch_labels = []
		self.sequence_lens = []
		self.masks = []

		keys = sorted(features.keys())
		for f in keys:

			# Get the feature using the id as a key
			feature = features[f]

			# Ignore examples that fall out of the specified range
			if feature.shape[1] > self.max_input or len(labels[f]) > self.max_output: 
				continue

			# Append padding to input features, and get sequence length
			feature, sequence_len = self.pad_feature(feature)

			# Append padding to the output
			label, mask = self.pad_label(labels[f])
			self.input_ids.append(f)
			self.batch_features.append(feature)
			self.batch_labels.append(label)
			self.sequence_lens.append(sequence_len)
			self.masks.append(mask)

		# Convert lists to numpy arrays
		self.input_ids = np.array(self.input_ids)

		self.batch_labels = np.array(self.batch_labels)

		self.batch_features = np.array(self.batch_features)
		# Normalize features if you want
		if self.normalize:
			if self.mean_vector is None:
				self.mean_vector = np.mean(np.mean(self.batch_features, axis=2), axis=0)
			self.batch_features = (self.batch_features.transpose((0,2,1)) - self.mean_vector).transpose((0,2,1))
			
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


	def get_batch(self, batch_size=32, shuffle=True):
	    rand_indices = np.random.choice(range(self.num_examples), size=batch_size, replace=False)
	    return self.batch_features[rand_indices], self.sequence_lens[rand_indices], self.batch_labels[rand_indices], \
	    		self.masks[rand_indices]

	def decode(self, input_seq):
		output = ''
		for val in input_seq:
			next_char = self.ints_to_chars[val]
			output += self.ints_to_chars[val]
			if val == self.end_char:
				break
		return output


