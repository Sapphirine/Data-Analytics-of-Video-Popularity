# Search video reference and channel information on Twitter
# Created by Ziyu He

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time
import json
import urllib2
import oauth2
import threading

#Twitter API consumer key, consumer secret, access token, access token secret
L_consumer_key = ["Your Consumer Key List"]
L_consumer_secret = ["Your Consumer Secret List"]
L_access_token = ["Your Access Token List"]
L_access_token_secret = ["Your Access Token Secret List"]



def oauth_req(url, consumer_key, consumer_secret, token_key, token_secret, http_method="GET", post_body="", http_headers=None):
    consumer = oauth2.Consumer(key=consumer_key, secret=consumer_secret)
    token = oauth2.Token(key=token_key, secret=token_secret)
    client = oauth2.Client(consumer, token)
    resp, content = client.request( url, method=http_method, body=post_body, headers=http_headers )
    return content


def test():
    res = []
    for item in L:
        url = "https://api.twitter.com/1.1/search/tweets.json?f=tweets&vertical=default&q=" + item + "&src=typd"
        C = oauth_req(url, access_token, access_token_secret)
        res.append(C)
    return res


#Each thread handles a number of videos
#collecting strategy: query 180 times every 15 mins, when we get rate exceeding message, wait for 10 seconds
class tw_search(threading.Thread):
    def __init__(self, consumer_key, consumer_secret, token_key, token_secret, threadID, videoIDlist):
        threading.Thread.__init__(self)
        self.ckey = consumer_key
        self.csecret = consumer_secret
        self.tkey = token_key
        self.tsecret = token_secret
        self.threadID = threadID
        self.videoIDs = videoIDlist
    def run(self):
        videos = self.videoIDs
        threadID = self.threadID
        print "Starting--" + str(threadID)
        count = 0
        data = {}
        for v in videos:
            if count == 0:
                start = time.time()
            if count <= 150:
                url = "https://api.twitter.com/1.1/search/tweets.json?f=tweets&vertical=default&q=" + str(v) + "&src=typd"
                content = oauth_req(url, self.ckey, self.csecret, self.tkey, self.tsecret)
                if content.find("Rate limit exceeded") != -1:
                    time.sleep(10)
                    content = oauth_req(url, self.ckey, self.csecret, self.tkey, self.tsecret)
                data[v] = content
                #with open('twitter_raw_data/twitter_raw_data'+str(threadID)+'.txt', 'a') as f:
                with open('twitter_raw_data/twitter_raw_data_miss2'+str(threadID)+'.txt', 'a') as f:
                    f.write(str({v:data[v]})+'\n')
                count = count + 1
            if count > 150:
                count = 0
                wait = 900 - (time.time()-start)
                print str(threadID) + ":--start waiting " + str(wait)
                time.sleep(wait)
                # while (time.time() - start)<=900.0:
                #     pass
                print str(threadID) + ":--the wait is over"

#Collect info of referrers of all the videos we would like to query
def collect(vmapN):
    N = len(vmapN)
    videos = [[vmapN[i][key] for key in vmapN[i]] for i in range(N)]
    threads = []
    try:
        for i in range(N):
            thread = tw_search(L_consumer_key[i], L_consumer_secret[i], L_access_token[i], L_access_token_secret[i], "Thread--"+str(i), videos[i])
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    except:
        print "Error: unable to start thread"


#After the first round of collecting from Twiter, check how many queries are failed due to (1)Rate Exceeding (2)Invalid keywords
#Collect videoids correspond to these failed queries
def check1():
    L = []
    for i in range(8):
        with open("twitter_raw_data/twitter_raw_dataThread--"+str(i)+".txt") as f:
            L = L + f.readlines()
    L = [eval(item.rstrip()) for item in L]
    D = {}
    for item in L:
        D[item.keys()[0]] = item[item.keys()[0]]
    miss1 = []
    miss2 = []
    for key in D:
        if D[key].find('Rate limit exceeded') != -1:
            miss1.append(key)
        if key[0] == '-':
            miss2.append(key[1:])
    return(miss1, miss2)


#Collect for the second round: the previous failed queries due to invalid keyword or rate exceeding
def collect_miss(miss):
    #misslist = [miss[i*230:(i+1)*230] for i in range(8)]
    misslist = [miss[i*165:(i+1)*165] for i in range(8)]
    threads = []
    N = len(misslist)
    try:
        for i in range(N):
            thread = tw_search(L_consumer_key[i], L_consumer_secret[i], L_access_token[i], L_access_token_secret[i], "Thread--"+str(i), misslist[i])
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    except:
       print "Error: unable to start thread"

#Parse video data into N parts
def parse(V,N):
    vmap = dict(zip(range(len(V)),[item['videoId'] for item in V]))
    vmapN = []
    for i in range(N):
        tmp = {}
        if i != N-1:
            for j in range(len(V))[i*10350:(i+1)*10350]:
                tmp[j] = vmap[j]
        else:
            for j in range(len(V))[i*10350:]:
                tmp[j] = vmap[j]
        vmapN.append(tmp)
    return vmapN



#After the 2nd round of collecting data, combine all the data collected in 2 rounds
def combine():
    L = []
    for i in range(8):
        with open("twitter_raw_data/twitter_raw_dataThread--"+str(i)+".txt") as f:
            L = L + f.readlines()
    L = [eval(item.rstrip()) for item in L] 
    D = {}
    for item in L:
        D[item.keys()[0]] = item[item.keys()[0]]
    L1 = []
    for i in range(8):
        with open("twitter_raw_data/twitter_raw_data_missThread--"+str(i)+".txt") as f:
            L1 = L1 + f.readlines()
    L1 = [eval(item.rstrip()) for item in L1]
    L2 = []
    for i in range(8):
        with open("twitter_raw_data/twitter_raw_data_miss2Thread--"+str(i)+".txt") as f:
            L2 = L2 + f.readlines()
    L2 = [eval(item.rstrip()) for item in L2]   
    for item in L1:
        D[item.keys()[0]] = item[item.keys()[0]]
    for item in L2:
        D[item.keys()[0]] = item[item.keys()[0]]    
    return D

#Summarize the followers for each referrer of each viedeo of each channel, and obtain the mapping
def check2(tweets):
    count = 0
    true = True
    false = False
    null = None
    Vref = {}
    for key in tweets:
        ref_follow = {}
        if tweets[key].find('"statuses":[],"search_metadata"') != -1:
            Vref[key] = {}
        else:
            try:
                tweet = eval(tweets[key])["statuses"]
                for item in tweet:
                    ref_follow[item['user']['name']] = item['user']['followers_count']
            except:
                pass
            Vref[key] = ref_follow
    Vref = [Vref[key] for key in Vref if key[0] != '-']
    return Vref



#Collect follower numbers of channels on Twitter
def find_account(accounts):
    i = 0
    try:
        thread = tw_channel(L_consumer_key[i], L_consumer_secret[i], L_access_token[i], L_access_token_secret[i], "Thread--"+str(i), accounts)
        thread.start()
    except:
        print "Error: unable to start thread" 



#Collect follower numbers of channels on Twitter
class tw_channel(threading.Thread):
    def __init__(self, consumer_key, consumer_secret, token_key, token_secret, threadID, accounts):
        threading.Thread.__init__(self)
        self.ckey = consumer_key
        self.csecret = consumer_secret
        self.tkey = token_key
        self.tsecret = token_secret
        self.threadID = threadID
        self.accounts = accounts
    def run(self):
        count = 0
        for channel in accounts:
            if count == 0:
                start = time.time()  
            if count <= 150:         
                url = "https://api.twitter.com/1.1/users/search.json?q=" + str(accounts[channel]) + "Transfermarkt&page=1&count=3"
                content = oauth_req(url, self.ckey, self.csecret, self.tkey, self.tsecret)
                if content.find("Rate limit exceeded") != -1:
                    time.sleep(20)
                    content = oauth_req(url, self.ckey, self.csecret, self.tkey, self.tsecret)
                with open('twitter_raw_data/channelfollowers.txt', 'a') as f:
                    num =  float(eval(content)[0]['followers_count'])/1000.0
                    f.write(str({channel:num})+'\n')
            if count > 150:
                count = 0
                wait = 900 - (time.time()-start)
                print str(threadID) + ":--start waiting " + str(wait)
                time.sleep(wait)
                print str(threadID) + ":--the wait is over"


#Process the result from find_account()
def check3(C):
    with open('twitter_raw_data/channelfollowers.txt', 'r') as f:
        data = f.readlines()
    dic = {channel:0.0 for channel in C}
    for item in data:
        dic[eval(item).keys()[0]] = eval(item)[eval(item).keys()[0]]
    followers = np.matrix([dic[channel] for channel in C])
    return followers


 
