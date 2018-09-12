from __future__ import division
import cPickle as pickle
import csv
import numpy as np
import sys
import os
import cPickle as pickle
from nltk.corpus import stopwords
import json
import gzip
from tqdm import tqdm
from collections import Counter
from collections import defaultdict


def batchify(data, i, bsz, max_sample):
    start = int(i * bsz)
    end = int(i * bsz) + bsz
    if(end>max_sample):
        end = max_sample
    data = data[start:end]
    return data

def dict_to_list(data_dict):
    data_list = []
    for key, value in tqdm(data_dict.items(),
                            desc='dict conversion'):
        for v in value:
            data_list.append([key, v[0], v[1]])
    return data_list

def dictToFile(dict,path):
    print "Writing to {}".format(path)
    with gzip.open(path, 'w') as f:
        f.write(json.dumps(dict))

def dictFromFileUnicode(path):
    '''
    Read js file:
    key ->  unicode keys
    string values -> unicode value
    '''
    print "Loading {}".format(path)
    with gzip.open(path, 'r') as f:
        return json.loads(f.read())

def get_single_word_overlap_features(q, a, word2df, stopwords=None):
    if(stopwords is None):
        stopwords = set()
    q_set = set([x for x in q if x not in stopwords])
    a_set = set([x for x in a if x not in stopwords])
    word_overlap = q_set.intersection(a_set)

    overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))
    word_overlap = q_set.intersection(a_set)
    df_overlap = 0.0

    for w in word_overlap:
      df_overlap += word2df[w]

    return [overlap, df_overlap]

def add_overlap_features(X1, X2, word2df):
    print("Adding overlap features...")
    print("Loaded word2df")
    stoplist = set(stopwords.words('english'))
    print("Loaded {} stop words".format(len(stoplist)))
    pairs = zip(X1,X2)
    all_feature_vecs = []
    for p in pairs:
        feature_vec = []
        q, a = p[0],p[1]
        non_stop_words_feat = get_single_word_overlap_features(q, a, word2df, stopwords=None)
        stop_words_feat = get_single_word_overlap_features(q, a, word2df, stopwords=stoplist)
        feature_vec += non_stop_words_feat
        feature_vec += stop_words_feat
        all_feature_vecs.append(feature_vec)
    return all_feature_vecs

def add_overlap_features_single(q, a, word2df, stoplist=None):
    # print("Adding overlap features...")
    # print("Loaded word2df")

    # print("Loaded {} stop words".format(len(stoplist)))
    feature_vec = []
    # q, a = p[0],p[1]
    non_stop_words_feat = get_single_word_overlap_features(q, a, word2df, stopwords=None)
    stop_words_feat = get_single_word_overlap_features(q, a, word2df, stopwords=stoplist)
    feature_vec += non_stop_words_feat
    feature_vec += stop_words_feat
   # print(feature_vec)
    return feature_vec

def load_pickle(fin):
    with open(fin,'r') as f:
        obj = pickle.load(f)
    return obj

def select_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    if(gpu>=0):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

def load_pickle(fin):
    with open(fin,'r') as f:
        obj = pickle.load(f)
    return obj

def load_set(fin):
    data = []
    with open(fin, 'r') as f:
        reader= csv.reader(f, delimiter='\t')
        for r in reader:
            data.append(r)
    return data

def question2id(q, questions):
    qids = []
    for w in q:
        if(w in word_index):
            qids.append(word_index[w])
        else:
            # unknown token
            qids.append(2)
    return qids

def triple_convert(t, questions):
    try:
        q1 = questions[int(t[0])]
        q2 = questions[int(t[1])]
        label = int(t[2])
    except:
        return None
    if(len(q1)==0 or len(q2)==0):
        return None
    return [q1,q2,label]

def build_set(fin, questions):
    data = load_set(fin)
    data = [triple_convert(x, questions) for x in data]
    data = [x for x in data if x is not None]
    return data

def one_hot_label(x):
    # Simple convertor for binary classification
    if(x==1):
        return [0,1]
    elif(x==0):
        return [1,0]

def length_stats(lengths, name=''):
    print("=====================================")
    print("Length Statistics for {}".format(name))
    print("Max={}".format(np.max(lengths)))
    print("Median={}".format(np.median(lengths)))
    print("Mean={}".format(np.mean(lengths)))
    print("Min={}".format(np.min(lengths)))

def show_stats(name, x):
    print("{} max={} mean={} min={}".format(name,
                                        np.max(x),
                                        np.mean(x),
                                        np.min(x)))

def print_args(args, path=None):
    if path:
        output_file = open(path, 'w')
    args.command = ' '.join(sys.argv)
    items = vars(args)
    if path:
        output_file.write('=============================================== \n')
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    if path:
        output_file.write('=============================================== \n')
    if path:
        output_file.close()
    del args.command

def mkdir_p(path):
    if path == '':
        return
    try:
        os.makedirs(path)
    except:
        pass
