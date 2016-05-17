#Named Entity Recognition
#Created by Ziyu He

import nltk 
import numpy as np

def extract_entity_names(t):
    entity_names = []
    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))
    return entity_names

def name_rec1(sample):
    sentences = nltk.sent_tokenize(sample)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))
    return entity_names

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
