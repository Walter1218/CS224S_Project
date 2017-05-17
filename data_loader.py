import pickle
# Load minibatch function

#input
#labels
#mask
#seq

class DataLoader:

	def __init__(self, dataset, max_input, max_output, normalize=False, split='train'):
		# Load the data from pickle file depending on input
		self.num_train_examples = 10000
		self.max_time = 1929
		self.max_char = 336
		self.max_input = max_input
		self.max_output = max_output
		self.normalize = normalize
		self.split = split
		self.pad_char = 27
		self.start_char = 28
		self.end_char = 29

		pathname = 'data/'+dataset+'/'
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
		for f in features.keys():
			feature = features[f]
			if feature.shape[1] > self.max_input or len(label[f]) > self.max_output: continue
			# Normalize features if you want
			if self.normalize: feature = normalize(feature)

			# Append start, end, padding tokens
			feature, sequence_len = pad_feature(self, feature)
			label, mask = pad_label(self, labels[f])
			self.input_ids.append(f)
			self.batch_features.append(feature)
			self.batch_labels.append(label)
			self.sequence_lens.append(sequence_lens)
			self.masks.append(mask)

	
	def normalize(feature):
		return feature


	def pad_feature(self, feature):
		pad = np.zeros((feature.shape[0], self.max_input - feature.shape[1]))
		if self.feature_pad == 'front':
			return np.transpose(np.append(pad, feature, 1)), feature.shape[1]
		else:
			return np.transpose(np.append(feature, pad, 1)), feature.shape[1]

	def pad_label(self, label):
		pad = [self.pad_char]*(self.max_char - len(label))
		if self.label_pad == 'front':
			mask = [0] * len(pad) + [1] * (len(label)+2)
			return pad + [self.start_char] + feature + [self.end_char], mask
		else:
			mask = [1] * (len(label)+2)+[0] * len(pad)
			return [self.start_char] + feature + [self.end_char] + pad, mask


	def get_minibatch(self, shuffle=True):
	    """
	    Iterates through the provided data one minibatch at at time. You can use this function to
	    iterate through data in minibatches as follows:
	        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
	            ...
	    Or with multiple data sources:
	        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
	            ...
	    Args:
	        data: there are two possible values:
	            - a list or numpy array
	            - a list where each element is either a list or numpy array
	        minibatch_size: the maximum number of items in a minibatch
	        shuffle: whether to randomize the order of returned data
	    Returns:
	        minibatches: the return value depends on data:
	            - If data is a list/array it yields the next minibatch of data.
	            - If data a list of lists/arrays it returns the next minibatch of each element in the
	              list. This can be used to iterate through multiple data sources
	              (e.g., features and labels) at the same time.
	    """
	    data = [self.batch_features, self.sequence_lens, self.batch_labels, self.masks]
	    minibatch_size = self.minibatch_size
	    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
	    data_size = len(data[0]) if list_data else len(data)
	    indices = np.arange(data_size)
	    if shuffle:
	        np.random.shuffle(indices)
	    for minibatch_start in np.arange(0, data_size, minibatch_size):
	        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
	        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
	            else minibatch(data, minibatch_indices)


	def minibatch(data, minibatch_idx):
	    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

	def minibatches(data, batch_size, shuffle=True):
	    batches = [np.array(col) for col in zip(*data)]
	    return get_minibatches(batches, batch_size, shuffle)