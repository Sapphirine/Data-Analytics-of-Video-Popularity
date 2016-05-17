#Training, Validation, Testing of Random Forest model
#Create by Ziyu He

import json
import numpy as np
import datetime as dt
import urllib
import oauth2
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
#from gensim import corpora, models
#import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from NE import name_rec1
from operator import itemgetter


#Train a single Random Forest classifier
def Train1(X, y):
	rfc = RandomForestClassifier(n_estimators=10, oob_score=True)
	rfc.n_classes_ = 3
	model = rfc.fit(X, y)
	return model


#Train a cascaded Random Forest classifier
#In each level discard class 1 1data which is classified correctly
def cascadedRF_train(X, y, N):
	models = []
	for level in range(N):
		row = X.shape[0]
		model = Train1(X, y)
		models.append(model)
		k = 0
		for i in range(row):
			i = i - k
			if model.predict(X[i,:]) == y[i] and y[i] == '1':
				X = np.delete(X, (i), axis=0)
				y = np.delete(y, (i))
				k = k + 1
	return models


#Use cascaded Random Forest classifier to predict
def cascadedRF_predict(data, models):
	pred = []
	row = data.shape[0]
	for i in range(row):
		point = data[i,:]
		for model in models:
			if model.predict(point) == '1':
				pred.append('1')
				break
			if models.index(model) == len(models) - 1:
				pred.append(model.predict(point))
	return pred


#Test Single RF
def test(model, testdata, testy):
	pred = model.predict(testdata)
	count = 0
	#AB: predicted as "1" while really is "2"
	AA, AB, AC, BA, BB, BC, CA, CB, CC = 0,0,0,0,0,0,0,0,0
	TA, TB, TC, PA, PB, PC = 0,0,0,0,0,0
	for i in range(testy.shape[0]):
		if pred[i] == '1' and testy[i] == '1':
			AA = AA + 1
			PA = PA + 1
			TA = TA + 1
		if pred[i] == '1' and testy[i] == '2':
			AB = AB + 1
			PA = PA + 1
			TB = TB + 1
		if pred[i] == '1' and testy[i] == '3':
			AC = AC + 1
			PA = PA + 1
			TC = TC + 1
		if pred[i] == '2' and testy[i] == '1':
			BA = BA + 1	
			PB = PB + 1
			TA = TA + 1
		if pred[i] == '2' and testy[i] == '2':
			BB = BB + 1
			PB = PB + 1
			TB = TB + 1
		if pred[i] == '2' and testy[i] == '3':
			BC = BC + 1
			PB = PB + 1
			TC = TC + 1
		if pred[i] == '3' and testy[i] == '1':
			CA = CA + 1
			PC = PC + 1
			TA = TA + 1	
		if pred[i] == '3' and testy[i] == '2':
			CB = CB + 1
			PC = PC + 1
			TB = TB + 1
		if pred[i] == '3' and testy[i] == '3':
			CC = CC + 1
			PC = PC + 1
			TC = TC + 1		
		if pred[i] != testy[i]:
			count = count + 1
	Aerr = (AB + AC)/float(PA)
	Berr = (BA + BC)/float(PB)
	Cerr = (CA + CB)/float(PC)
	Err = count/float(testy.shape[0])
	return (Aerr, Berr, Cerr, Err)


#Test Cascaded RF
def test_cascaded(models, testdata, testy):
	N = len(models)
	err = {"Model "+str(i):test(models[i], testdata, testy) for i in range(N)}
	pred = cascadedRF_predict(testdata, models)
	count = 0
	#AB: predicted as "1" while really is "2"
	AA, AB, AC, BA, BB, BC, CA, CB, CC = 0,0,0,0,0,0,0,0,0
	TA, TB, TC, PA, PB, PC = 0,0,0,0,0,0
	for i in range(testy.shape[0]):
		if pred[i] == '1' and testy[i] == '1':
			AA = AA + 1
			PA = PA + 1
			TA = TA + 1
		if pred[i] == '1' and testy[i] == '2':
			AB = AB + 1
			PA = PA + 1
			TB = TB + 1
		if pred[i] == '1' and testy[i] == '3':
			AC = AC + 1
			PA = PA + 1
			TC = TC + 1
		if pred[i] == '2' and testy[i] == '1':
			BA = BA + 1	
			PB = PB + 1
			TA = TA + 1
		if pred[i] == '2' and testy[i] == '2':
			BB = BB + 1
			PB = PB + 1
			TB = TB + 1
		if pred[i] == '2' and testy[i] == '3':
			BC = BC + 1
			PB = PB + 1
			TC = TC + 1
		if pred[i] == '3' and testy[i] == '1':
			CA = CA + 1
			PC = PC + 1
			TA = TA + 1	
		if pred[i] == '3' and testy[i] == '2':
			CB = CB + 1
			PC = PC + 1
			TB = TB + 1
		if pred[i] == '3' and testy[i] == '3':
			CC = CC + 1
			PC = PC + 1
			TC = TC + 1		
		if pred[i] != testy[i]:
			count = count + 1
	Aerr = (AB + AC)/float(PA)
	Berr = (BA + AC)/float(PB)
	Cerr = (CA + CB)/float(PC)
	Err = count/float(testy.shape[0])
	err['Overall'] = (Aerr, Berr, Cerr, Err)
	return err



#K-Fold cross validation of model
def Train_Kfold(X, y, K):
	#y = np.array(y)
	kf = KFold(X.shape[0], n_folds = K)

	record = {}

	k = 0
	for train_index, test_index in kf:
		k = k + 1
		rfc = RandomForestClassifier(n_estimators=5, oob_score=True)
		rfc.n_classes_ = 3
		model = rfc.fit(X[train_index], y[train_index])
		pred = model.predict(X[test_index])
		count = 0
		#AB: predicted as "1" while really is "2"
		AA, AB, AC, BA, BB, BC, CA, CB, CC = 0,0,0,0,0,0,0,0,0
		TA, TB, TC, PA, PB, PC = 0,0,0,0,0,0
		for i in range(len(pred)):
			if pred[i] == '1' and y[test_index][i] == '1':
				AA = AA + 1
				PA = PA + 1
				TA = TA + 1
			if pred[i] == '1' and y[test_index][i] == '2':
				AB = AB + 1
				PA = PA + 1
				TB = TB + 1
			if pred[i] == '1' and y[test_index][i] == '3':
				AC = AC + 1
				PA = PA + 1
				TC = TC + 1
			if pred[i] == '2' and y[test_index][i] == '1':
				BA = BA + 1	
				PB = PB + 1
				TA = TA + 1
			if pred[i] == '2' and y[test_index][i] == '2':
				BB = BB + 1
				PB = PB + 1
				TB = TB + 1
			if pred[i] == '2' and y[test_index][i] == '3':
				BC = BC + 1
				PB = PB + 1
				TC = TC + 1
			if pred[i] == '3' and y[test_index][i] == '1':
				CA = CA + 1
				PC = PC + 1
				TA = TA + 1	
			if pred[i] == '3' and y[test_index][i] == '2':
				CB = CB + 1
				PC = PC + 1
				TB = TB + 1
			if pred[i] == '3' and y[test_index][i] == '3':
				CC = CC + 1
				PC = PC + 1
				TC = TC + 1		
			if pred[i] != y[test_index][i]:
				count = count + 1
		record[str(k)] = [count, AA, AB, AC, BA, BB, BC, CA, CB, CC, TA, TB, TC, PA, PB, PC, len(pred)]
	err, Aerr, Berr, Cerr = 0, 0, 0, 0
	for key in record:
		Aerr = Aerr + (record[key][2]+record[key][3])/float(record[key][13])
		Berr = Berr + (record[key][4]+record[key][5])/float(record[key][14])
		#Cerr = Cerr + (record[key][7]+record[key][8])/float(record[key][15])
		err = err + record[key][0]/float(record[key][16])
	err = err/float(K)
	Aerr = err/float(K)
	Berr = err/float(K)
	#Cerr = err/float(K)
	#err = float(count)/K
	#AA, AB, AC, BA, BB, BC, CA, CB, CC = float(AA)/K, float(AB)/K, float(AC)/K, float(BA)/K, float(BB)/K, float(BC)/K, float(CA)/K, float(CB)/K, float(CC)/K
	return (err, Aerr, Berr)


#Get the rank of features' relative importance given by Random Forest model
def feature_rank(model):
	f = np.ndarray.tolist(model.feature_importances_)
	fmap = {str(i):f[i] for i in range(len(f))}
	rank = []
	ff = sorted(f)
	ff.reverse()
	for item in ff:
		if item != 0:
			for key in fmap:
				if fmap[key] == item:
					rank.append(key)
	return rank

#Item Based Recommender
def recommender(outdict, preference):
	data = outdict
	scores_mapping = {key:0.0 for key in data if key not in preference.keys()}
	for key in scores_mapping:
		for key1 in preference:
			scores_mapping[key] = scores_mapping[key] + float(preference[key1])*np.array(data[key]).dot(np.array(data[key1]))
	top5 = sorted(scores_mapping.items(), key=itemgetter(1))[:5]
	return top5




