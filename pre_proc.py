import glob2
import numpy as np
import librosa
import pickle
import re

SEED = 472
PCT_TRAIN =.70
PCT_DEV = .15
PCT_TEST = .15
SAMPLE_RATE = 16000
N_MELS = 40
hop_length = 160
n_mfcc = 39


def split_and_save_data(dataset):

	dirname1 = "data\\wsj0_raw_data\\**\\wsj0\\s*_tr_*\\**\\*.wv1"
	dirname2 = "data\\wsj0_raw_data\\**\\wsj0\\s*_dt_*\\**\\*.wv1"
	dirname3 = "data\\wsj0_raw_data\\**\\wsj0\\s*_et_*\\**\\*.wv1"

	fp = glob2.glob(dirname1)+glob2.glob(dirname2)+glob2.glob(dirname3)

	unique_files = set()
	unique_fp = []
	for f in fp:
		filename = re.search('\w+.wv1', f).group(0).split('.')[0]
		if filename not in unique_files:
			unique_files.add(filename)
			unique_fp.append(f)
	fp = unique_fp

	num_files = len(fp)

	num_train = PCT_TRAIN*num_files
	num_dev = PCT_DEV*num_files
	num_test = num_files-num_train-num_dev

	mfcc_train = {}
	mfcc_dev = {}
	mfcc_test = {}
	fp_to_id = {}
	id_to_fp = {}

	print "Splitting Files and Features..."
	np.random.seed(SEED)
	idxs = np.arange(num_files)
	np.random.shuffle(idxs)
	data_split = {'train':idxs[:int(num_files*PCT_TRAIN)], 'dev':idxs[int(num_files*PCT_TRAIN):int(num_files*(PCT_TRAIN+PCT_DEV))], 'test':idxs[int(num_files*(PCT_TRAIN+PCT_DEV)):]}

	print "Done Processing Files!"
	print "Total Files Processed:", len(fp)
	print "Train Files:", len(data_split['train'])
	print "Dev Files:", len(data_split['dev'])
	print "Test Files:", len(data_split['test'])

	for i,f in enumerate(fp):
		if i%100==0: print i
		filename = re.search('\w+.wv1', f).group(0).split('.')[0]
		fp_to_id[filename] = i
		id_to_fp[i] = filename
		if i in data_split['train']:
			y, sr = librosa.load(f, sr=SAMPLE_RATE)
			# mfcc_train[i] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
			mfcc_train[i] = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=N_MELS))
		elif i in data_split['dev']:
			y, sr = librosa.load(f, sr=SAMPLE_RATE)
			# mfcc_dev[i] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
			mfcc_dev[i] = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=N_MELS))
		else:
			y, sr = librosa.load(f, sr=SAMPLE_RATE)
			# mfcc_test[i] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
			mfcc_test[i] = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=N_MELS))

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

	print "All Done!"

	return data_split, fp_to_id

def split_and_save_transcripts(dataset, data_split, fp_to_id):

	dirname1 = "data\\wsj0_raw_data\\11-4.1\\wsj0\\transcrp\\dots\\**\\**\\*.dot"
	dirname2 = "data\\wsj0_raw_data\\**\\wsj0\\s*_et_*\\**\\*.dot"

	fp = glob2.glob(dirname1)+glob2.glob(dirname2)

	num_files = len(fp)

	labels_train = {}
	labels_dev = {}
	labels_test = {}

	no_id = {}

	comp_tr = set(data_split['train'])
	comp_dev = set(data_split['dev'])
	comp_ts = set(data_split['test'])

	print "Splitting Files and Features..."
	
	for i,f in enumerate(fp):
		if i%100==0: print (i)
		with open(f, 'rb') as doc:
			for transcript in doc:
				try:
					filename = re.findall(r'\([^\)\(]+\)',transcript)[-1]
					transcript = transcript.replace(filename, '')
					transcript = re.sub(r"(\[[^\]\[]]+\])|[^a-zA-Z ]", '', transcript)
					filename = filename[1:-1]

					if filename in fp_to_id:
						id_no = fp_to_id[filename]
					else:
						no_id[filename] = f
						print f, filename
						continue

					num_string = []
					for char in transcript:
						if char == ' ':
							num_string.append(26)
						else:
							num_string.append(ord(char.lower()) - ord('a'))	

					if id_no in data_split['train']:
						labels_train[id_no] = num_string
						if id_no in comp_tr: comp_tr.remove(id_no)
					elif id_no in data_split['dev']:
						labels_dev[id_no] = num_string
						if id_no in comp_dev: comp_dev.remove(id_no)
					else:
						labels_test[id_no] = num_string
						if id_no in comp_ts: comp_ts.remove(id_no)
				except:
					print transcript

	print "Done Processing Files!"
	print "Train Files:", len(labels_train)
	print "Dev Files:", len(labels_dev)
	print "Test Files:", len(labels_test)

	print "Writing Data to Files..."

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

	no_id_pkl = open('data\\'+dataset+'\\no_id.pkl', 'wb')
	pickle.dump(no_id, no_id_pkl)
	no_id_pkl.close()
	print "Label id's with no data saved to data\\"+dataset+"\\no_id.pkl"

	print "All Done!"


if __name__ == "__main__":
	dataset = 'wsj0'
	data_split, fp_to_id = split_and_save_data(dataset)
	# data_split = pickle.load(open('data\\wsj0\\data_split.pkl', 'rb'))
	# fp_to_id = pickle.load(open('data\\wsj0\\filepath_to_id_no.pkl', 'rb'))
	split_and_save_transcripts(dataset, data_split, fp_to_id)