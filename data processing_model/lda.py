#Training of LDA Model
#Create by Ziyu He

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from pyspark.mllib.clustering import LDA, LDAModel
#from pyspark.mllib.linalg import Vectors


def LDA():
	#Read Data
	with open('videodescription.txt', 'r') as V:
		V_des = eval(V.read())

	#Initialization
	tokenizer = RegexpTokenizer(r'\w+')
	en_stop = get_stop_words('en')
	p_stemmer = PorterStemmer()
	    
	#Aggregate descriptions
	unempty = []
	for key in V_des2:
		if V_des2[key] != u'':
			unempty.append(key)
	sample = range(0,len(unempty), 100)

	documents = []
	for i in sample:
		documents.append(V_des[unempty[i]])

	texts = []

	#Preprocessing
	for doc in documents:
	    raw = doc.lower()
	    tokens = tokenizer.tokenize(raw)
	    stopped_tokens = [doc for doc in tokens if not doc in en_stop]
	    stemmed_tokens = [p_stemmer.stem(doc) for doc in stopped_tokens]
	    texts.append(stemmed_tokens)

	rmlist = [u'http', u'youtub', u'youtube', u'www', u'com', u's', u'the', u'a', u'de', u'twitter', u'facebook', u'video', u'subscrib', u'instagram', u'channel',u't\xe0u', u'\xe2',]
	for item in rmlist:
		for line in texts:
			while item in line:
				line.remove(item)

	#Create Term Dictionary
	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]

	#Obtain our LDA model
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
	return ldamodel





#Spark Version of LDA
def LDA_spark():
	data = sc.textFile("data/mllib/sample_lda_data.txt")
	parsedData = data.map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
	corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

	# Cluster the documents into three topics using LDA
	Model = LDA.train(corpus, k=3)
			
	# Save and load model
	Model.save(sc, "myModelPath")
	sameModel = LDAModel.load(sc, "myModelPath")