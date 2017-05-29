import glob2
import numpy as np
import librosa
import pickle
import re

SEED = 472
PCT_TRAIN =.70
PCT_DEV = .15
PCT_TEST = .15
hop_length = 512
n_mfcc = 39

NUM_TO_STR = {'z':'zero', 'o':'oh', '1':'one', '2':'two', '3':'three', '4':'four', '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine'}
NUM_TO_NUM = {'z':0, 'o':10, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}

def split_and_save_data(dataset):

	dirname = "data\\tidigits_raw_data\\**\\tidigits\\t*\\**\\**\\*.wav"
	fp = glob2.glob(dirname)

	num_files = len(fp)

	num_train = PCT_TRAIN*num_files
	num_dev = PCT_DEV*num_files
	num_test = num_files-num_train-num_dev

	mfcc_train = {}
	mfcc_dev = {}
	mfcc_test = {}
	fp_to_id = {}
	id_to_fp = {}
	labels_train = {}
	labels_dev = {}
	labels_test = {}

	print "Splitting Files and Features..."
	np.random.seed(SEED)
	idxs = np.arange(num_files)
	np.random.shuffle(idxs)
	data_split = {'train':idxs[:int(num_files*PCT_TRAIN)], 'dev':idxs[int(num_files*PCT_TRAIN):int(num_files*(PCT_TRAIN+PCT_DEV))], 'test':idxs[int(num_files*(PCT_TRAIN+PCT_DEV)):]}

	#data_split = pickle.load(open("data\\tidigits\\data_split.pkl", 'rb'))
	print "Done Processing Files!"
	print "Total Files Processed:", len(fp)
	print "Train Files:", len(data_split['train'])
	print "Dev Files:", len(data_split['dev'])
	print "Test Files:", len(data_split['test'])

	for i,f in enumerate(fp):
		if i%100==0: print i
		filename = f
		label = parse_label(re.search('\w+.wav', f).group(0).split('.')[0][:-1])
		label = parse_label(re.search('\w+.wav', f).group(0).split('.')[0][:-1])
		fp_to_id[filename] = i
		id_to_fp[i] = filename
		if i in data_split['train']:
			y, sr = librosa.load(f)
			mfcc_train[i] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
			labels_train[i] = label
		elif i in data_split['dev']:
			y, sr = librosa.load(f)
			mfcc_dev[i] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
			labels_dev[i] = label
		else:
			y, sr = librosa.load(f)
			mfcc_test[i] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
			labels_test[i] = label

	print "Done Processing Files!"
	print "Total Files Processed:", len(fp)
	print "Train Files:", len(data_split['train'])
	print "Dev Files:", len(data_split['dev'])
	print "Test Files:", len(data_split['test'])

	print "Writing Data to Files..."

	data_split_pkl = open('data\\'+dataset+'\\data_split.pkl', 'wb')
	pickle.dump(data_split, data_split_pkl)
	data_split_pkl.close()
	print "Data splits saved to data\\"+dataset+"\\data_split.pkl"

	fp_to_id_pkl = open('data\\'+dataset+'\\filepath_to_id_no.pkl', 'wb')
	pickle.dump(fp_to_id, fp_to_id_pkl)
	fp_to_id_pkl.close()
	print "Map of filepath names to id number saved to data\\"+dataset+"\\filepath_to_id_no.pkl"

	id_to_fp_pkl = open('data\\'+dataset+'\\id_no_to_filepath.pkl', 'wb')
	pickle.dump(id_to_fp, id_to_fp_pkl)
	id_to_fp_pkl.close()
	print "Map of id number to filepath name saved to data\\"+dataset+"\\id_no_to_filepath.pkl"

	mfcc_train_pkl = open('data\\'+dataset+'\\mfcc_train.pkl', 'wb')
	pickle.dump(mfcc_train, mfcc_train_pkl)
	mfcc_train_pkl.close()
	print "Train set MFCC features saved to data\\"+dataset+"\\mfcc_train.pkl"

	mfcc_dev_pkl = open('data\\'+dataset+'\\mfcc_dev.pkl', 'wb')
	pickle.dump(mfcc_dev, mfcc_dev_pkl)
	mfcc_dev_pkl.close()
	print "Dev set MFCC features saved to data\\"+dataset+"\\mfcc_dev.pkl"

	mfcc_test_pkl = open('data\\'+dataset+'\\mfcc_test.pkl', 'wb')
	pickle.dump(mfcc_test, mfcc_test_pkl)
	mfcc_test_pkl.close()
	print "Test set MFCC features saved to data\\"+dataset+"\\mfcc_test.pkl"

	labels_train_pkl = open('data\\'+dataset+'\\labels_train.pkl', 'wb')
	pickle.dump(labels_train, labels_train_pkl)
	labels_train_pkl.close()
	print "Train set labels features saved to data\\"+dataset+"\\labels_train.pkl"

	labels_dev_pkl = open('data\\'+dataset+'\\labels_dev.pkl', 'wb')
	pickle.dump(labels_dev, labels_dev_pkl)
	labels_dev_pkl.close()
	print "Dev set labels features saved to data\\"+dataset+"\\labels_dev.pkl"

	labels_test_pkl = open('data\\'+dataset+'\\labels_test.pkl', 'wb')
	pickle.dump(labels_test, labels_test_pkl)
	labels_test_pkl.close()
	print "Test set labels features saved to data\\"+dataset+"\\labels_test.pkl"

	print "All Done!"

	return data_split, fp_to_id

'''
#version to change to character form
def parse_label(label):
	result = []
	for i in range(len(label)):
		result.append(NUM_TO_STR[label[i]])
	result = " ".join(result)
	num_string = []
	for char in result:
		if char == ' ':
			num_string.append(26)
		else:
			num_string.append(ord(char.lower()) - ord('a'))
	return num_string
'''
#version to keep numerical form
def parse_label(label):
	result = []
	for i in range(len(label)):
		result.append(NUM_TO_NUM[label[i]])
	return result


if __name__ == "__main__":
	dataset = 'tidigits'
	data_split, fp_to_id = split_and_save_data(dataset)