# Process Data
# Created by Ziyu He

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
import twitter_search as tw
from NE import name_rec1


#Read channel and video data from JSON files
def read_data():
	V = []
	for i in range(10):
		for j in range(9):
			try:
				with open("video_data/VideoData"+str(i)+"part"+str(j)+".json") as f:
					V = V + json.load(f)
			except:
				break

	with open('ChannelData.json') as cdata:
		Cdata = json.load(cdata)

	return (V, Cdata)


#Clean data
#there are 10 videos withour stat and snippet(i.e. no description, channelId, title...etc), we have to discard theses
#there are 8298 videos without tags (they all have snippet), we can keep them
def clean_data(V):
	stat, comm, snippet, des, pubtime, tags, notag, catId, title, chId = [], [], [], [], [], [], [], [], [], []
	for i in range(len(V)):
		item = V[i]
		v_id = item['videoId']
		if item['statistics'] == None:
			stat.append(v_id)
		if item['comments'] == None:
			comm.append(v_id)
		try:
			if item['snippet']['description'] == None:
				des.append(v_id)
			if item['snippet']['publishedAt'] == None:
				pubtime.append(v_id)
			try:
				if item['snippet']['tags'] == None:
					tags.append(v_id)
			except:
				notag.append(v_id)
			if item['snippet']['categoryId'] == None:
				catId.append(v_id)
			if item['snippet']['title'] == None:
				title.append(v_id)
			if item['snippet']['channelId'] == None:
				chId.append(v_id)
		except:
			snippet.append(i)
	#Discard all the videos without snippet
	for i in range(len(snippet)):
		V.pop(snippet[i]-i)

	return V



#Get the mapping of channel to video
#Get the info of videos
def video_sum(V):
	V_stat, V_comm, V_des, V_pubtime, V_tags, V_catId, V_title, VC_dict = {}, {}, {}, {}, {}, {}, {}, {}
	for item in V:
		v_id = item['videoId']
		V_stat[v_id] = item['statistics']
		V_comm[v_id] = item['comments']
		V_des[v_id] = item['snippet']['description']
		V_pubtime[v_id] = item['snippet']['publishedAt']
		try:
			V_tags[v_id] = item['snippet']['tags']
		except:
			V_tags[v_id] = None
		V_catId[v_id] = item['snippet']['categoryId']
		V_title[v_id] = item['snippet']['title']
		if not VC_dict.has_key(item['snippet']['channelId']):
			VC_dict[item['snippet']['channelId']] = [v_id]
		if VC_dict.has_key(item['snippet']['channelId']):
			VC_dict[item['snippet']['channelId']].append(v_id)

	catId_set = set([V_catId[key] for key in V_catId])

	return (V_stat, V_comm, V_des, V_pubtime, V_tags, V_catId, V_title, VC_dict, catId_set)


#Get info of sampled channels
def channel_sum(Cdata, VC_dict):
	C_tmp = {}
	for item in Cdata:
		c_id = item['channelID']
		if not C_tmp.has_key(c_id):
			C_tmp[c_id] = item
	C, C_stat, C_des, C_title, C_pubtime = {}, {}, {}, {}, {}
	misschannel = []
	for key in VC_dict:
		try:
			channel = C_tmp[key]
			C[key] = channel
			C_stat[key], C_des[key], C_title[key], C_pubtime[key] = channel['statistics'], channel['snippet']['description'], channel['snippet']['title'], channel['snippet']['publishedAt']
		except:
			misschannel.append(key)
			pass
	for channel in misschannel:
		del VC_dict[channel]
	return (C, C_stat, C_des, C_title, C_pubtime)



#Apply KMeans clustering on channel data and return statistics of clustering result, K is number of clusters
def Clustering(C, K):
	# commentCount = [int(item["statistics"][u'commentCount']) for item in C]
	# viewCount = [int(item["statistics"][u'viewCount']) for item in C]
	# videoCount = [int(item["statistics"][u'videoCount']) for item in C]
	# subCount = [int(item["statistics"][u'subscriberCount']) for item in C]
	commentCount = [int(C[item]["statistics"][u'commentCount']) for item in C]
	viewCount = [int(C[item]["statistics"][u'viewCount']) for item in C]
	videoCount = [int(C[item]["statistics"][u'videoCount']) for item in C]
	subCount = [int(C[item]["statistics"][u'subscriberCount']) for item in C]

	mat = np.matrix
	X = mat([commentCount, viewCount, videoCount, subCount]).T

	#KMeans
	random_state = 170
	pred = KMeans(n_clusters=K, random_state=random_state).fit_predict(X)

	#Compute statistics for each cluster
	clusterdata = {str(i):[] for i in range(K)}
	clusterrange = {str(i):0 for i in range(K)}
	clustermean = {str(i):0 for i in range(K)}
	clusterstd = {str(i):0 for i in range(K)}
	clusternumber = {str(i):0 for i in range(K)}

	for i in range(pred.shape[0]):
		for label in range(K):
			if pred[i] == label:
				clusterdata[str(label)].append(X[i, 1])
				break

	for label in range(K):
		clusterrange[str(label)] = (min(clusterdata[str(label)]), max(clusterdata[str(label)]))
		clustermean[str(label)] = np.array(clusterdata[str(label)]).mean()
		clusterstd[str(label)] = np.array(clusterdata[str(label)]).std()
		clusternumber[str(label)] = len(clusterdata[str(label)])

	return (clusterdata, clusterrange, clustermean, clusterstd, clusternumber)


#Applying KMeans clustering assuming cluster number is 3
def Clustering3(C):
	commentCount = [int(item["statistics"][u'commentCount']) for item in C]
	viewCount = [int(item["statistics"][u'viewCount']) for item in C]
	videoCount = [int(item["statistics"][u'videoCount']) for item in C]
	subCount = [int(item["statistics"][u'subscriberCount']) for item in C]

	mat = np.matrix
	X = mat([commentCount, viewCount, videoCount, subCount]).T

	#KMeans
	random_state = 170
	pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

	#Compute statistics for each cluster
	l0, l1, l2, idx0, dx1, idx2 = [], [], [], [], [], []
	for i in range(pred.shape[0]):
		if pred[i] == 0:
			l0.append(X[i,1])
			idx0.append(i)
		if pred[i] == 1:
			l1.append(X[i,1])
			idx1.append(i)
		if pred[i] == 2:
			l2.append(X[i,1])
			idx2.append(i)
	Range0 = (min(l0), max(l0))
	Range1 = (min(l0), max(l0))
	Range2 = (min(l0), max(l0))
	mean0, mean1, mean2 = np.array(l0).mean(), np.array(l1).mean(), np.array(l2).mean()
	std0, std1, std2 = np.array(l0).std(), np.array(l1).std(), np.array(l2).std()
	len0, len1, len2 = len(l0), len(l1), len(l2)
	return (Range0, Range1, Range2, mean0, mean1, mean2, std0, std1, std2, len0, len1, len2)



#Get the PCA representation of quantitation of channel popularity
#Apply on clustering result and plot clusters
def PCA(C_stat, C):
	C_view, C_video, C_sub, C_comm, pred = [], [], [], [], []
	for key in C_stat:
		C_view.append(int(C_stat[key][u'viewCount']))
		C_video.append(int(C_stat[key][u'videoCount']))
		C_sub.append(int(C_stat[key][u'subscriberCount']))
		C_comm.append(int(C_stat[key][u'commentCount']))
		if int(C_stat[key][u'viewCount']) < 850000000:
			pred.append(1)
		elif 850000000 <= int(C_stat[key][u'viewCount']) < 3700000000:
			pred.append(2)
		else:
			pred.append(3)
	#Standardization
	C_view_s, C_video_s, C_sub_s, C_comm_s = [], [], [], []
	viewmean, viewstd = np.array(C_view).mean(), np.array(C_view).std()
	videomean, videostd = np.array(C_video).mean(), np.array(C_video).std()
	submean, substd = np.array(C_sub).mean(), np.array(C_sub).std()
	commmean, commstd = np.array(C_comm).mean(), np.array(C_comm).std()
	for i in range(len(C_view)):
		C_view_s.append((C_view[i]-viewmean)/viewstd)
		C_video_s.append((C_video[i]-videomean)/videostd)
		C_sub_s.append((C_sub[i]-submean)/substd)
		C_comm_s.append((C_comm[i]-commmean)/commstd)

	Pmatrix = np.matrix([C_view_s, C_video_s, C_sub_s, C_comm_s]).T
	pca = PCA(n_components=2)
	pca.fit(Pmatrix)
	Xpca = pca.transform(Pmatrix)

	plt.scatter(Xpca[:, 0], Xpca[:, 1], c = pred)
	plt.savefig('pcachannelstandard.jpg')

	i = 0
	pcadict = {}
	for key in C:
		pcadict[key] =  np.ndarray.tolist(Xpca[i,:])
		i = i + 1

	with open("Xpca.txt", "a") as f:
		f.write(str(pcadict))
	return Xpca


#Generate part of channel features including (as matrix mat):
	#Duration, video publishing frequency
	#Ratio of subscriber/view, comment/view
	#Sentiment anaylysis result: positive/negative/neutral/compound sentiment scores
#Generate response vector y according to clustering result:
	#cluster1: [0, 850000000], cluster2:[850000000, 3700000000], cluster3[3700000000, infinity]
def Cgen(C, C_stat, C_des, C_title, C_pubtime, V_stat, V_comm, VC_dict):
	T, v_freq, sub_view, comm_view, comm_sentiment_Neg, comm_sentiment_Neu, comm_sentiment_Pos, comm_sentiment_Comp = [], [], [], [], [], [], [], []
	y = []
	keys = C.keys()
	end = dt.datetime.now()
	for key in C:
		#Get the time of each channel
		t = str(C_pubtime[key])
		start = dt.datetime(year = int(t[:4]), month = int(t[5:7]), day = int(t[8:10]), hour = int(t[11:13]), minute = int(t[14:16]), second = int(t[17:19]), microsecond = int(t[20:23]))
		T.append((end - start).days)
		#Get the video publishing frequency of each channel
		v_freq.append(float(C_stat[key]['videoCount'])/float((end - start).days))
		#Get the subscriber/viewer and comment/viewer ratio
		if float(C_stat[key]['subscriberCount']) == 0 or float(C_stat[key]['viewCount']) == 0 or float(C_stat[key]['commentCount']) == 0:
			sub_view.append(0)
			comm_view.append(0)
		else:
			sub_view.append(float(C_stat[key]['subscriberCount'])/float(C_stat[key]['viewCount']))
			comm_view.append(float(C_stat[key]['commentCount'])/float(C_stat[key]['viewCount']))
		#Process the comments of videos of each channel
		sid = SentimentIntensityAnalyzer()
		cs = np.matrix([0.0,0.0,0.0,0.0])
		for video in VC_dict[key]:
			if V_comm[video] != []:
				vs = np.matrix([0.0,0.0,0.0,0.0])
				for sentence in V_comm[video]:
					score = sid.polarity_scores(sentence)
					vs[0,0] = vs[0,0] + score['neg']
					vs[0,1] = vs[0,1] + score['neu']
					vs[0,2] = vs[0,2] + score['pos']
					vs[0,3] = vs[0,3] + score['compound']
				cs = cs + float(V_stat[video][u'viewCount']) * vs/float(len(V_comm[video]))
		cs = cs/float(len(VC_dict[key]))
		comm_sentiment_Neg.append(cs[0,0])
		comm_sentiment_Neu.append(cs[0,1])
		comm_sentiment_Pos.append(cs[0,2])
		comm_sentiment_Comp.append(cs[0,3])
		#Get response as categorical labels
		if float(C_stat[key]['viewCount']) < 850000000:
			y.append('1')
		elif 850000000 <= float(C_stat[key]['viewCount']) < 3700000000:
			y.append('2')
		else:
			y.append('3')
	mat = np.matrix([T, v_freq, sub_view, comm_view, comm_sentiment_Neg, comm_sentiment_Neu, comm_sentiment_Pos, comm_sentiment_Comp])
	return (np.array(y), mat)


#Aggregate the video statistics for each channel including:
	#Ratio of like/dislike, comment/view, favorite/view
def Vgen(V_stat, VC_dict):
	like_dislike, comm_view, fav_view = [], [], []
	for video in V_stat:
		if not V_stat[video].has_key(u'dislikeCount'):
			V_stat[video][u'dislikeCount'] = u'0'
		if not V_stat[video].has_key(u'likeCount'):
			V_stat[video][u'likeCount'] = u'0'
		if not V_stat[video].has_key(u'favouriteCount'):
			V_stat[video][u'favouriteCount'] = u'0'
		if V_stat[video][u'viewCount'] == u'0':
			V_stat[video][u'viewCount'] = u'1000000000000000000000000000'
	for key in VC_dict:
		vlist = VC_dict[key]
		tmp = 0
		for video in vlist:
			if int(V_stat[video]['dislikeCount']) == 0:
				tmp = tmp + float(V_stat[video]['likeCount'])
			else:
				tmp = tmp + float(V_stat[video]['likeCount'])/float(V_stat[video]['dislikeCount'])
		like_dislike.append(tmp/float(len(vlist)))
		#slike_dislike.append(sum([float(V_stat[item]['likeCount'])/float(V_stat[item]['dislikeCount']) for item in vlist])/float(len(vlist)))
		comm_view.append(sum([float(float(V_stat[item]['commentCount'])/float(V_stat[item]['viewCount'])) for item in vlist])/float(len(vlist)))
		fav_view.append(sum([float(float(V_stat[item]['favouriteCount'])/float(V_stat[item]['viewCount'])) for item in vlist])/float(len(vlist)))
	V_agg = np.matrix([like_dislike, comm_view, fav_view])
	return V_agg



#Process and aggregate the category info of videos of a channel as dummy variable (44 categorys, 43 dummy variables)
def cat_dummy(VC_dict, V_catId, V_stat):
	cat_id_dummy = []
	for channel in VC_dict:
		video_cat = [0 for i in range(43)]
		for video in VC_dict[channel]:
			if int(V_catId[video])!=44:
				video_cat[int(V_catId[video])-1] = video_cat[int(V_catId[video])-1] + 1
		cat_id_dummy.append([item/float(len(VC_dict[channel])) for item in video_cat])
	cat_id_dummy = np.matrix(cat_id_dummy).T
	return cat_id_dummy 


#Compute the referrer scores for each channel: sum(sum of each video's referrer's followers/1000.0)
def ref_scores():
	tweets = tw.combine()
	Vref = tw.check2(tweets)
	Vref1 = {channel:0 for channel in VC_dict}
	for channel in VC_dict:
		for key in VC_dict[channel]:
			try:
				if key[0] == '-':
					key = key[1:]
				if Vref[key] != {}:
					Vref1[channel] = Vref1[channel] + sum([Vref[key][k]/1000.0 for k in Vref[key]])
			except:
				pass
	Ref_scores = np.matrix([Vref1[key] for key in Vref1])
	return Ref_scores


#Identify if channel has twitter, facebook, instagram google+
def socialmedia(C):
	score = {channel:0 for channel in C}
	for channel in C:
		if C[channel]['snippet']['description'].find('twitter')!=-1 or C[channel]['snippet']['description'].find('Twitter')!=-1:
			score[channel] = score[channel] + 1
		if C[channel]['snippet']['description'].find('facebook')!=-1 or C[channel]['snippet']['description'].find('Facebook')!=-1:
			score[channel] = score[channel] + 1
		if C[channel]['snippet']['description'].find('instagram')!=-1 or C[channel]['snippet']['description'].find('Instagram')!=-1:
			score[channel] = score[channel] + 1	
		if C[channel]['snippet']['description'].find('google')!=-1 or C[channel]['snippet']['description'].find('Google')!=-1:
			score[channel] = score[channel] + 1
	social = np.matrix([score[channel] for channel in C])
	return social

#get the channels with twitter accounts:
def get_account(C):
	accounts = {}
	for channel in C:
		tmp = C[channel]['snippet']['description'].find('twitter')
		if tmp != -1:
			accounts[channel] = C[channel]['snippet']['description'][tmp:tmp+50]
	return accounts


#Named Entity Recognition: compute the average occuring times of named entities in description and titles for each video of each channel
def C_names(VC_dict, V_des, V_title):
    C_name_map = {}
    C_name = []
    for channel in VC_dict:
        videos = VC_dict[channel]
        num = 0
        for video in videos:
            num = num + len(name_rec1(V_des[video])) + len(name_rec1(V_title[video]))
        ave = num/float(len(videos))
        C_name_map[channel] = ave
        C_name.append(ave)
    C_name = np.matrix(C_name)
    return (C_name_map, C_name)


#Aggregate all the features: 
	#C_name is score of named entity recognition
	#cat_id_dummy are ecoded dummy features of video categories
	#Ref_scores are aggregated referrer scores
	#Social media scores
def Agg1(mat, V_agg, C_name, Ref_scores, cat_id_dummy, social, y, C):
	X = np.concatenate((mat, V_agg, C_name, Ref_scores, cat_id_dummy, social))
	i = 0
	outdict, ydict = {}, {}
	for key in C:
		outdict[key] = np.ndarray.tolist(X[i,:])[0]
		ydict[key] = y[i]
		i = i + 1
	with open("features.txt", "a") as f:
		f.write(str(outdict))
	with open("response.txt", "a") as ff:
		ff.write(str(ydict))
	return (X.T, y)



#Aggregate all the processors
def PROCESSER():
	(V, Cdata) = read_data()
	V = clean_data(V)
	(V_stat, V_comm, V_des, V_pubtime, V_tags, V_catId, V_title, VC_dict, catId_set) = video_sum(V)
	(C, C_stat, C_des, C_title, C_pubtime) = channel_sum(Cdata)
	(y, mat) = Cgen(C, C_stat, C_des, C_title, C_pubtime, V_stat, V_comm, VC_dict)
	V_agg = Vgen(V_stat, VC_dict)
	cat_id_dummy = cat_dummy(VC_dict, V_catId, V_stat)
	Ref_scores = ref_scores()
	social = socialmedia(C)
	(C_name_map, C_name) = C_names(VC_dict, V_des, V_title)
	(X, y) = Agg1(mat, V_agg, C_name, Ref_scores, cat_id_dummy, social, y)
	return (X, y)





