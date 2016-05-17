from flask import Flask
from flask import render_template
from pymongo import MongoClient
from flask import request
import json
from bson import json_util
from bson.json_util import dumps
from flask import redirect
import math

app = Flask(__name__)

MONGODB_HOST = '0.0.0.0'
MONGODB_PORT = 27017
DBS_NAME = 'bigdata'
connection = MongoClient(MONGODB_HOST, MONGODB_PORT)

@app.route("/")
@app.route("/index")
@app.route("/index.html")
def index():
    return redirect("/static/index.html")

@app.route("/visualization")
@app.route("/visual")
def visual():
    return redirect("/static/visual.html")


@app.route("/recommendation")
def recommendation():
    return redirect("/static/recommendation.html")


def distance(r,cdata):
    distance = 0
    weight = [5,4,3,2,1]
    fields = [u'sentiment', u'stov', u'cname', u'time', u'ctov', u'ref', u'freq']
    for i,c in enumerate(cdata):
        sum_square = 0
        for f in fields:
            sum_square += (c[f] - r[f])^2
        distance += math.sqrt(sum_square)*weight[i]
    return distance

def get_recommend(channels):
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    channeldata_collection = connection[DBS_NAME]['channeldata']
    c_ids = []
    c_datas = []
    for c in channels:
        data = channeldata_collection.find_one({'snippet.title':c})
        c_ids.append(data['channelID'])
            
    xmatrix_collection = connection[DBS_NAME]['xmatrix']
    for c in channels:
        data = xmatrix_collection.find_one({'id':c})
        c_ids.append(data)
    cursor = xmatrix_collection.find()
    to_recommend = []
    distances = []
    for r in cursor:
        dis = distance(r,c_datas)
        if len(to_recommend) <5:
            to_recommend.append(r['id'])
            distances.append(dis)
        else:
            if dis < max(distances):
                ind = distances.index(max(distances))
                to_recommend[ind] = r['id']
                distances[ind] = dis
    print "to recommend!!!!!"
    print to_recommend
    connection.close()
    return to_recommend

def get_channel_record(channelID):
    connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
    channeldata_collection = connection[DBS_NAME]['channeldata']
    data = channeldata_collection.find_one({'channelID':channelID})
    connection.close()
    return data


@app.route("/recommend")
def recommend():
    channels = []
    channels.append(request.args.get('channel1'))
    channels.append(request.args.get('channel2'))
    channels.append(request.args.get('channel3'))
    channels.append(request.args.get('channel4'))
    channels.append(request.args.get('channel5')) 
    print channels

    to_recommend = get_recommend(channels)
    channel_infos = []
    for r in to_recommend:
        channel_infos.append(get_channel_record(r))
    json_result = json.dumps(channel_infos, default=json_util.default)
    return json_result

@app.route('/get_channel_info/<channelid>')
def get_channel_info(channelid=None):
    if channelid!= None:
        connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
        channeldata_collection = connection[DBS_NAME]['channeldata']
        xmatrix_collection = connection[DBS_NAME]['xmatrix']
        cursor = channeldata_collection.find({'channelID':channelid})
        result = []
        for r in cursor:
            result.append(r)
        cursor = xmatrix_collection.find({'id':channelid})
        for r in cursor:
            result.append(r)
        json_result = json.dumps(result, default=json_util.default)
        connection.close()
        return json_result

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80,debug=True)
#    titles = [u'Alltime10s', u'Ashish Singhal', u'DRAGUNOV911', u'Kingrich Media', u'BSG', u'PointlessBlogVlogs', u'JSConf', u'\u041a\u0430\u043d\u0430\u043b \u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044f SuperBCMC', u'Pine Dragon', u'LMFAOVEVO', u'Jaballa Hasan', u'Razieme Iborra', u'Happy Kids TV', u'jacksepticeye', u'kaysee clifford', u'College Basketball Player Highlights', u'\u0645\u062f\u0648\u0646\u0629 \u0627\u0644\u062a\u0627\u0633\u0641\u0631\u0647 \u0627\u0644\u0645\u0648\u0631\u064a\u062a\u0627\u0646\u064a\u0629', u'Shane Stroup', u'Goa Mattes', u'ThisIsCochise', u'American Vegan Society']
#    selected = titles[0:5]
#    get_recommend(selected)
