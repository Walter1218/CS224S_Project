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


def split_and_save_data(dataset):

	dirname_clean = "data\\LDC2017S07\\CHiME2_Grid\\data\\chime2-grid\\train\\clean\\**\\*.wav"
	dirname_emb = "data\\LDC2017S07\\CHiME2_Grid\\data\\chime2-grid\\train\\embedded\\\**\\*.wav"
	dirname_iso = "data\\LDC2017S07\\CHiME2_Grid\\data\\chime2-grid\\train\\isolated\\**\\*.wav"
	dirname_rev = "data\\LDC2017S07\\CHiME2_Grid\\data\\chime2-grid\\train\\reverberated\\**\\*.wav"

	fp_clean = glob2.glob(dirname_clean)
	fp_emb=glob2.glob(dirname_emb)
	fp_iso=glob2.glob(dirname_iso)
	fp_rev=glob2.glob(dirname_rev)

	fps = [fp_clean, fp_emb, fp_iso, fp_rev]

	num_files = len(fp_clean)

	num_train = PCT_TRAIN*num_files
	num_dev = PCT_DEV*num_files
	num_test = num_files-num_train-num_dev

	mfcc_train_clean = {}
	mfcc_dev_clean = {}
	mfcc_test_clean = {}
	mfcc_train_iso = {}
	mfcc_dev_iso = {}
	mfcc_test_iso = {}
	mfcc_train_emb = {}
	mfcc_dev_emb = {}
	mfcc_test_emb = {}
	mfcc_train_rev = {}
	mfcc_dev_rev = {}
	mfcc_test_rev = {}
	fp_to_id = {}
	id_to_fp = {}

	print "Splitting Files and Features..."
	np.random.seed(SEED)
	idxs = np.arange(num_files)
	np.random.shuffle(idxs)
	data_split = {'train':idxs[:int(num_files*PCT_TRAIN)], 'dev':idxs[int(num_files*PCT_TRAIN):int(num_files*(PCT_TRAIN+PCT_DEV))], 'test':idxs[int(num_files*(PCT_TRAIN+PCT_DEV)):]}

	print "Done Processing Files!"
	print "Total Files Processed:", len(fp_clean)
	print "Train Files:", len(data_split['train'])
	print "Dev Files:", len(data_split['dev'])
	print "Test Files:", len(data_split['test'])

	for j,fp in enumerate(fps):
		for i,f in enumerate(fp):
			if i%100==0: print (i)
			filename = re.search('\w+.wav', f).group(0).split('.')[0]
			if j == 0:
				fp_to_id[filename] = i
				id_to_fp[i] = filename
				fid = i
			else:
				fid = fp_to_id[filename]
			if fid in data_split['train']:
				y, sr = librosa.load(f)
				if j==0:
					mfcc_train_clean[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
				elif j==1:
					mfcc_train_emb[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
				elif j==2:
					mfcc_train_iso[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
				else:
					mfcc_train_rev[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
			elif fid in data_split['dev']:
				y, sr = librosa.load(f)
				if j==0:
					mfcc_dev_clean[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
				elif j==1:
					mfcc_dev_emb[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
				elif j==2:
					mfcc_dev_iso[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
				else:
					mfcc_dev_rev[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
			else:
				y, sr = librosa.load(f)
				if j==0:
					mfcc_test_clean[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
				elif j==1:
					mfcc_test_emb[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
				elif j==2:
					mfcc_test_iso[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
				else:
					mfcc_test_rev[fid] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)

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

	mfcc_train_clean_pkl = open('data\\'+dataset+'\\mfcc_train_clean.pkl', 'wb')
	pickle.dump(mfcc_train_clean, mfcc_train_clean_pkl)
	mfcc_train_clean_pkl.close()
	print "Train set MFCC features saved to data\\"+dataset+"\\mfcc_train_clean.pkl"

	mfcc_dev_clean_pkl = open('data\\'+dataset+'\\mfcc_dev_clean.pkl', 'wb')
	pickle.dump(mfcc_dev_clean, mfcc_dev_clean_pkl)
	mfcc_dev_clean_pkl.close()
	print "Dev set MFCC features saved to data\\"+dataset+"\\mfcc_dev_clean.pkl"

	mfcc_test_clean_pkl = open('data\\'+dataset+'\\mfcc_test_clean.pkl', 'wb')
	pickle.dump(mfcc_test_clean, mfcc_test_clean_pkl)
	mfcc_test_clean_pkl.close()
	print "Test set MFCC features saved to data\\"+dataset+"\\mfcc_test_clean.pkl"

	mfcc_train_emb_pkl = open('data\\'+dataset+'\\mfcc_train_emb.pkl', 'wb')
	pickle.dump(mfcc_train_emb, mfcc_train_emb_pkl)
	mfcc_train_emb_pkl.close()
	print "Train set MFCC features saved to data\\"+dataset+"\\mfcc_train_emb.pkl"

	mfcc_dev_emb_pkl = open('data\\'+dataset+'\\mfcc_dev_emb.pkl', 'wb')
	pickle.dump(mfcc_dev_emb, mfcc_dev_emb_pkl)
	mfcc_dev_emb_pkl.close()
	print "Dev set MFCC features saved to data\\"+dataset+"\\mfcc_dev_emb.pkl"

	mfcc_test_emb_pkl = open('data\\'+dataset+'\\mfcc_test_emb.pkl', 'wb')
	pickle.dump(mfcc_test_emb, mfcc_test_emb_pkl)
	mfcc_test_emb_pkl.close()
	print "Test set MFCC features saved to data\\"+dataset+"\\mfcc_test_emb.pkl"

	mfcc_train_iso_pkl = open('data\\'+dataset+'\\mfcc_train_iso.pkl', 'wb')
	pickle.dump(mfcc_train_iso, mfcc_train_iso_pkl)
	mfcc_train_iso_pkl.close()
	print "Train set MFCC features saved to data\\"+dataset+"\\mfcc_train_iso.pkl"

	mfcc_dev_iso_pkl = open('data\\'+dataset+'\\mfcc_dev_iso.pkl', 'wb')
	pickle.dump(mfcc_dev_iso, mfcc_dev_iso_pkl)
	mfcc_dev_iso_pkl.close()
	print "Dev set MFCC features saved to data\\"+dataset+"\\mfcc_dev_iso.pkl"

	mfcc_test_iso_pkl = open('data\\'+dataset+'\\mfcc_test_iso.pkl', 'wb')
	pickle.dump(mfcc_test_iso, mfcc_test_iso_pkl)
	mfcc_test_iso_pkl.close()
	print "Test set MFCC features saved to data\\"+dataset+"\\mfcc_test_iso.pkl"

	mfcc_train_rev_pkl = open('data\\'+dataset+'\\mfcc_train_rev.pkl', 'wb')
	pickle.dump(mfcc_train_rev, mfcc_train_rev_pkl)
	mfcc_train_rev_pkl.close()
	print "Train set MFCC features saved to data\\"+dataset+"\\mfcc_train_rev.pkl"

	mfcc_dev_rev_pkl = open('data\\'+dataset+'\\mfcc_dev_rev.pkl', 'wb')
	pickle.dump(mfcc_dev_rev, mfcc_dev_rev_pkl)
	mfcc_dev_rev_pkl.close()
	print "Dev set MFCC features saved to data\\"+dataset+"\\mfcc_dev_rev.pkl"

	mfcc_test_rev_pkl = open('data\\'+dataset+'\\mfcc_test_rev.pkl', 'wb')
	pickle.dump(mfcc_test_rev, mfcc_test_rev_pkl)
	mfcc_test_rev_pkl.close()
	print "Test set MFCC features saved to data\\"+dataset+"\\mfcc_test_rev.pkl"

	print "All Done!"

	return data_split, fp_to_id

def split_and_save_transcripts(dataset, data_split, fp_to_id):

	dirname = "data\\LDC2017S07\\CHiME2_Grid\\data\\eval_tools_grid\\labels\\allids.mlf"

	labels_train = {}
	labels_dev = {}
	labels_test = {}

	print "Splitting Files and Features..."

	with(open(dirname, 'rb')) as f:
		line = f.readline()
		while True:
			line = f.readline()#for line in f:
			if not line: break
			filename = re.search('\w+.lab', line).group(0).split('.')[0]
			label = []
			while True:
				line = f.readline().strip()
				if line == ".": break
				label.append(line)
			label = ' '.join(label)

			if filename in fp_to_id:
				id_no = fp_to_id[filename]
			else:
				print filename
				continue

			num_string = []
			for char in label:
				if char == ' ':
					num_string.append(26)
				else:
					num_string.append(ord(char.lower()) - ord('a'))	

			if id_no in data_split['train']:
				labels_train[id_no] = num_string
			elif id_no in data_split['dev']:
				labels_dev[id_no] = num_string
			else:
				labels_test[id_no] = num_string
	

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

	print "All Done!"


if __name__ == "__main__":
	dataset = 'chime2_grid'
	data_split, fp_to_id = split_and_save_data(dataset)
	#data_split = pickle.load(open('data\\chime2_grid\\data_split.pkl', 'rb'))
	#fp_to_id = pickle.load(open('data\\chime2_grid\\filepath_to_id_no.pkl', 'rb'))

	split_and_save_transcripts(dataset, data_split, fp_to_id)