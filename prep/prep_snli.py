from __future__ import print_function
from functools import reduce
import json
import os
import re
import tarfile
import tempfile
import numpy as np
import keras
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import np_utils
from tqdm import tqdm
import cPickle as pickle
from utilities import *
import gzip
from collections import Counter
from nltk.stem.porter import PorterStemmer
import re

import nltk

min_count = 0
porter_stemmer = PorterStemmer()
tokenizer = nltk.tokenize.TreebankWordTokenizer()

dataset = "SNLI"
fp = './datasets/SNLI'

def extract_tokens_from_binary_parse(parse):
    parse = parse.replace('(', ' ')
    parse = parse.replace(')', ' ')
    parse = parse.replace('-LRB-', '(').replace('-RRB-', ')')
    return parse.split()

def parse_pos_tag(parse):
  base_parse = [s.rstrip(" ").rstrip(")") for s in parse.split("(") if ")" in s]
  pos = [pair.split(" ")[0] for pair in base_parse]
  return pos

def yield_examples(fn, skip_no_majority=True, limit=None):
  for i, line in enumerate(open(fn)):
    if limit and i > limit:
      break
    data = json.loads(line)
    label = data['gold_label']
    _s1 = extract_tokens_from_binary_parse(
                      data['sentence1_binary_parse'])
    _s2 = extract_tokens_from_binary_parse(
                      data['sentence2_binary_parse'])
    s1 = ' '.join(_s1)
    s2 = ' '.join(_s2)

    p1 = parse_pos_tag(data['sentence1_parse'])
    p2 = parse_pos_tag(data['sentence2_parse'])

    assert(len(p1)==len(_s1))
    assert(len(p2)==len(_s2))

    if skip_no_majority and label == '-':
      continue
    yield (label, s1, s2, p1, p2)

def get_data(fn, limit=None):
  raw_data = list(yield_examples(fn=fn, limit=limit))
  #aw_data = [x for x in raw_data]
  left = [s1 for _, s1, s2, _, _ in raw_data]
  right = [s2 for _, s1, s2, _, _ in raw_data]
  left = [s.encode('ascii') for s in left]
  right = [s.encode('ascii') for s in right]

  p1 = [p1 for _, _, _, p1, _ in raw_data]
  p2 = [p2 for _, _, _, _, p2 in raw_data]

  LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
  Y = [LABELS[l] for l, s1, s2, _, _ in raw_data]
  return left, right, Y, p1, p2


print("Loading data...")
training = get_data('../tf_snli/dataset/snli_1.0_train.jsonl')
validation = get_data('../tf_snli/dataset/snli_1.0_dev.jsonl')
test = get_data('../tf_snli/dataset/snli_1.0_test.jsonl')

print("Tokenizing..")

LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
USE_GLOVE = True

def to_tokens(x):
    # print(x)
    y = ['<S>'] + [x for x in x.split(' ')] + ['<E>']
    return y

def add_start_end(x):
    return ['<S>'] + x + ["<E>"]

def sent_to_char(instance):
  # print(instance)
  return [[y for y in x] for x in instance]

def prepare_data(data, set_type):
    print("Preparing..{}".format(set_type))
    left = [to_tokens(x) for x in tqdm(data[0])]
    right = [to_tokens(x) for x in tqdm(data[1])]

    p1 = data[3]
    p2 = data[4]

    p1 = [add_start_end(x) for x in p1]
    p2 = [add_start_end(x) for x in p2]

    # knowledge feats
    # k1, k2 = build_kb_feats(zip(left, right, p1, p2))
    k1, k2 = [0], [0]
    print("Computing EM features..")
    el1, el2 = build_em_feats(zip(left, right),
                              stem=True,
                              lower=True)

    left_chars = [sent_to_char(x) for x in tqdm(left)]
    right_chars = [sent_to_char(x) for x in tqdm(right)]
    labels = data[2]

    word_tokens = [left, right, labels]
    char_tokens = [left_chars, right_chars]
    features = [el1, el2]

    pos = [p1, p2]
    kg = [k1, k2]

    return word_tokens, char_tokens, features, pos, kg

def set_to_words(data=[]):
    words = []
    for d in data:
      for _d in d:
        words += _d
    return words

def set_to_chars(data=[]):
    chars = []
    for d in data:
      for _d in d:
        for __d in _d:
          chars += __d
    # print(chars)
    return chars

test, test_chars, test_feats, \
        test_pos, test_kg = prepare_data(test, 'test')
training, train_chars, train_feats, \
         train_pos, train_kg = prepare_data(training,'train')
validation, val_chars, val_feats, \
         val_pos, val_kg = prepare_data(validation, 'dev')


dev = validation

word_list = [training[0],training[1],
           dev[0], dev[1], test[1], test[0]]

char_list =  [train_chars[0],train_chars[1],
            val_chars[0], val_chars[1],
              test_chars[1], test_chars[0]]

pos_list = [train_pos[0], train_pos[1],
          test_pos[1], test_pos[0],
            val_pos[0], val_pos[1]]

words = set_to_words(data=word_list)
chars = set_to_chars(data=char_list)
pos = set_to_words(data=pos_list)

words = [x.lower() for x in words]

word_index, index_word = build_word_index(words,
                              min_count=min_count,
                              extra_words=['<pad>','<unk>'],
                              lower=False)

pos_index, pos_word = build_word_index(pos,
                            min_count=0,
                            extra_words=['<pad>'],
                            lower=False)

char_index, index_char = build_word_index(chars,
                            min_count=0,
                            extra_words=['<pad>'],
                            lower=False)

print(char_index.keys())
print(pos_index.keys())
if not os.path.exists(fp):
    os.makedirs(fp)


print('Build model...')
print('Vocab size = {}'.format(len(word_index)))
print("Char Size ={}".format(len(char_index)))
print("Pos Size={}".format(len(pos_index)))

training = zip(*training)
validation = zip(*validation)
test = zip(*test)

env = {
  'train':training,
  'dev':validation,
  'test':test,
  'word_index':word_index,
  'char_index':char_index,
  'pos_index':pos_index
}

feat_env = {
  'train_feats':zip(*train_feats),
  'dev_feats':zip(*val_feats),
  'test_feats':zip(*test_feats),
}

pos_env = {
  'train_pos':zip(*train_pos),
  'dev_pos':zip(*val_pos),
  'test_pos':zip(*test_pos)
}

kg_env = {
  'train_pos':zip(*train_kg),
  'dev_pos':zip(*val_kg),
  'test_pos':zip(*test_kg)
}


build_embeddings(word_index, index_word, out_dir=fp,
  init_type='zero', init_val=0.01,
  normalize=False)

print("Saved Glove")

dictToFile(env,'./datasets/{}/env.gz'.format(dataset))
dictToFile(feat_env,'./datasets/{}/feat_env.gz'.format(dataset))
dictToFile(pos_env,'./datasets/{}/pos_env.gz'.format(dataset))
dictToFile(kg_env,'./datasets/{}/kg_env.gz'.format(dataset))
