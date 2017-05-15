import glob2
import numpy as np
import librosa
import pickle

SEED = 472
PCT_TRAIN =.70
PCT_DEV = .15
PCT_TEST = .15
hop_length = 512
n_mfcc = 39


def split_and_save_data(dataset):

	if dataset == 'wsj0_si':
		dirname = "data\\wsj0_raw_data\\**\\wsj0\\si_tr_s\\**\\*.wv1"

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

	print ("Splitting Files and Features...")
	np.random.seed(SEED)
	idxs = np.arange(num_files)
	np.random.shuffle(idxs)
	data_split = {'train':idxs[:int(num_files*PCT_TRAIN)], 'dev':idxs[int(num_files*PCT_TRAIN):int(num_files*(PCT_TRAIN+PCT_DEV))], 'test':idxs[int(num_files*(PCT_TRAIN+PCT_DEV)):]}
	
	for i,f in enumerate(fp):
		if i%100==0: print (i)
		fp_to_id[f] = i
		id_to_fp[i] = f
		if i in data_split['train']:
			y, sr = librosa.load(f)
			mfcc_train[i] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
		elif i in data_split['dev']:
			y, sr = librosa.load(f)
			mfcc_dev[i] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
		else:
			y, sr = librosa.load(f)
			mfcc_test[i] = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)

	print ("Done Processing Files!")
	print ("Total Files Processed:", len(fp))
	print ("Train Files:", len(data_split['train']))
	print ("Dev Files:", len(data_split['dev']))
	print ("Test Files:", len(data_split['test']))

	print ("Writing Data to Files...")

	data_split_pkl = open('data_split.pkl', 'wb')
	pickle.dump(data_split, data_split_pkl)
	data_split_pkl.close()
	print ("Data splits saved to data_split.pkl")


	fp_to_id_pkl = open('data\\'+dataset+'\\filepath_to_id_no.pkl', 'wb')
	pickle.dump(fp_to_id, fp_to_id_pkl)
	fp_to_id_pkl.close()
	print ("Map of filepath names to id number saved to data\\"+dataset+"\\filepath_to_id_no.pkl")

	id_to_fp_pkl = open('data\\'+dataset+'\\id_no_to_filepath.pkl', 'wb')
	pickle.dump(id_to_fp, id_to_fp_pkl)
	id_to_fp_pkl.close()
	print ("Map of id number to filepath name saved to data\\"+dataset+"\\id_no_to_filepath.pkl")

	mfcc_train_pkl = open('data\\'+dataset+'\\mfcc_train.pkl', 'wb')
	pickle.dump(mfcc_train, mfcc_train_pkl)
	mfcc_train_pkl.close()
	print ("Train set MFCC features saved to data\\"+dataset+"\\mfcc_train.pkl")

	mfcc_dev_pkl = open('data\\'+dataset+'\\mfcc_dev.pkl', 'wb')
	pickle.dump(mfcc_dev, mfcc_dev_pkl)
	mfcc_dev_pkl.close()
	print ("Dev set MFCC features saved to data\\"+dataset+"\\mfcc_dev.pkl")

	mfcc_test_pkl = open('data\\'+dataset+'\\mfcc_test.pkl', 'wb')
	pickle.dump(mfcc_test, mfcc_test_pkl)
	mfcc_test_pkl.close()
	print ("Test set MFCC features saved to data\\"+dataset+"\\mfcc_test.pkl")

	print ("All Done!")

if __name__ == "__main__":
	dataset = 'wsj0_si'
	split_and_save_data(dataset)




	

