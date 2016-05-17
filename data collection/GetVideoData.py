# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 01:20:09 2016
This script run multiple threads to collect Youtube video data and channel data
@author: Haoxiang
"""

import os
import urllib2
import re
import json
import threading

from bs4 import BeautifulSoup
from apiclient.discovery import build
from apiclient.errors import HttpError

# Please ensure that you have enabled the YouTube Data API for your project.
DEVELOPER_KEY = "##########################"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def youtube_search_video(query,max_results=50):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)
  # Call the search.list method to retrieve results matching the specified
  # query term.
    search_response = youtube.search().list(q=query,part="id,snippet",maxResults=max_results).execute()
    videos = []
  # Add each result to the appropriate list, and then display the lists of
  # matching videos, channels, and playlists.
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            video_id = search_result["id"]["videoId"]
            channel_id = search_result["snippet"]["channelId"]
            videos.append({'videoID':video_id,'channelID':channel_id})        
        elif search_result["id"]["kind"] == "youtube#channel":
            pass
        elif search_result["id"]["kind"] == "youtube#playlist":
            pass
    return videos
    

def getLinks(page):
    Links = []
    soup = BeautifulSoup(page)
    soup = soup.find(id = "mw-content-text")
    
    for link in soup.find_all('a'):
        if link.has_attr('title'):
            href = link.get('href')
            Links.append(href)
    return(Links)

def getWikiTopics():
    strdomain = "https://en.wikipedia.org"
    StartPage = "/wiki/Portal:Contents/Portals"
    urlf = urllib2.urlopen(strdomain + StartPage)
        
    page = urlf.read()
    Links = getLinks(page)
    words=[]
    for link in Links:
        result = re.search(r'/wiki/Portal:(Contents/)*(\w+$)',link)
        if result!=None:
            word = result.group(2)
            words.append(word.replace("_"," "))
    print words
    return words
    
def downloadvideoinfo(words):
    videos = []
    for w in words:
        videos.extend(youtube_search_video(w))
    return videos


def get_video_stats(youtube,video_id):
    
    results = youtube.videos().list(
        part="statistics",
        id=video_id,
    ).execute()
    
    if len(results['items']) == 0:
        return None
    else:
        return results['items']
        
def get_video_snippet(youtube,video_id):
    
    results = youtube.videos().list(
        part="snippet",
        id=video_id,
    ).execute()
    if len(results['items']) == 0:
        return None
    else:
        return results['items']


def get_channel_videos(youtube, channel_id, maxresult=50):
    results = youtube.search().list(channelId=channel_id,type="video",part="id",maxResults = maxresult).execute()
    videos=[]
    for result in results['items']:
        v = result['id']['videoId']
        videos.append(v)
    return videos
    
def get_video_infos(youtube,video_id):
    results = youtube.videos().list(
        id=video_id,
        part="snippet,statistics",
        
    ).execute()
    if len(results['items']) == 0:
        return None
    else:
        return results['items'][0]

def get_video_comments(youtube,video_id):
    results = youtube.commentThreads().list(part="snippet",videoId=video_id,textFormat="plainText").execute()
    comments=[]
    for item in results["items"]:
        comment = item["snippet"]["topLevelComment"]
        text = comment["snippet"]["textDisplay"]
        comments.append(text)
    return comments

# a multi-threading function to collect all video data
def get_video_data_by_videoID_json(infile):
    with open(infile) as file:
        videos = json.load(file)
    n = 10
    width = len(videos)/n
    threads = []
    try:
        for i in xrange(n):
            thread = downloader(i,videos[(i*width):((i+1)*width)])
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
    except:
        print "Error: unable to start thread"
        
def get_channel_data(youtube, channel_id):
    results = youtube.channels().list(part="snippet,statistics",id=channel_id).execute()
    res = []
    for item in results["items"]:
        channel_data={}
        channel_data['channelID'] = channel_id
        channel_data['snippet'] = item["snippet"]
        channel_data['statistics'] = item["statistics"]
        res.append(channel_data)
    return res
        
def get_channel_data_by_channelID_json(infile,outfile,maxlength = 10000):
    with open(infile) as file:
        ChannelList = json.load(file)
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)
    ChannelData = []
    count = 0
    for c in ChannelList:
        cd = get_channel_data(youtube,c)
        ChannelData.extend(cd)
        count = count+1
        if count >=maxlength:
            break
        if count % 10 == 0:
            print str(count)+"channel done\n"
    with open(outfile,"w") as file:
        json.dump(ChannelData,file)
    

def get_video_by_channel(infile,output):
    with open(infile) as file:
        ChannelData = json.load(file) 
    ChannelList = [c['channelID'] for c in ChannelData]
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)
    videos = []
    length = 0
    for c in ChannelList:
        vs = get_channel_videos(youtube,c,maxresult = 10)
        videos.extend(vs)
        length = len(videos)
    with open(output,"w") as file:
        json.dump(videos,file)
        
def get_video_by_channelID(infile,output,maxlength = 100000):
    with open(infile) as file:
        ChannelData = json.load(file) 
    ChannelList = ChannelData
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)
    videos = []
    length = 0
    count = 0
    for c in ChannelList:
        vs = get_channel_videos(youtube,c,maxresult = 10)
        videos.extend(vs)
        length = len(videos)
        count = count+1
        if count % 10 == 0:
            print str(count)+"channel done\n"
        if length >= maxlength:
            break
    with open(output,"w") as file:
        json.dump(videos,file)

# a thread to download data for all videos
class downloader(threading.Thread):
    def __init__ (self, m1, m2):
        threading.Thread.__init__(self)
        self.m1 = m1
        self.m2 = m2
    def run(self):
        threadId =  self.m1
        videos = self.m2
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)
        VideoData = []    
        f = open('log'+str(threadId) +'.txt', 'w')
        count = 0
        for v in videos:
            try:
                data = {}
                data['videoId'] = v
                data['comments'] = get_video_comments(youtube,v)
                result = get_video_infos(youtube,v)
                if result != None:
                    data['snippet'] = result['snippet']
                    data['snippet'].pop('thumbnails',None)
                    data['statistics'] = result['statistics']
                else:
                    data['snippet'] = None
                    data['statistics'] = None
                VideoData.append(data)
                if len(VideoData) >= 1000:
                    with open('VideoData'+str(threadId)+'part'+str(count) +'.json',"w") as file:
                        json.dump(VideoData,file)
                    count = count + 1
                    VideoData = [] 
            except HttpError, e:
                print "An HTTP error"
                continue
            else:
                f.seek(0)
                f.write("thread"+str(threadId)+":" + str(len(VideoData)) + "video data collected")
        if len(VideoData) > 0:
            with open('VideoData'+str(threadId)+'part'+str(count) +'.json',"w") as file:
                json.dump(VideoData,file)
 
 
def get_search_words(infile):
    with open(infile) as f:
        lines = f.readlines()
        word_list = []
        for l in lines:
            words = l.split(',')
            for w in words:
                if w in word_list:
                    pass
                else:
                    word_list.append(unicode(w,'ascii','ignore'))
    return word_list

def merge_video_data_file(dirname):
    video_data=[]
    pattern = re.compile("VideoData[0-9]+part[0-9]+\.json$")
    for fname in os.listdir(dirname):
        if re.match(pattern,fname)!= None:
            with open(dirname + '/'+ fname) as f:
                data = json.load(f)
            video_data.extend(data)
    with open('VideoData.json','w') as f:
        json.dump(video_data,f)

if __name__ == "__main__":
    #extract random topics from wikipedia portal pages
    words = getWikiTopics()
    videos = downloadvideoinfo(words) 
    channels = [v['channelID'] for v in videos]
    #remove the duplicates
    channels = list(set(channels))
    #save the list of channel id as json file
    with open('ChannelIDList.json','w') as jsonfile:
        json.dump(channels,jsonfile)
    #call a function to get all video id that belong to the list of channel ID saved as json file
    get_video_by_channelID('ChannelIDList.json','VideosByChannel.json',10000)
    #call a function to get all video data from a file of the list of video ids
    get_video_data_by_videoID_json('VideosByChannel.json')
    #merge the video
    merge_video_data_file('videoData')
    get_channel_data_by_channelID_json('ChannelIDList.json','ChannelData.json',2000)
    




