from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .nn import *
from .compose_op import *
from .cnn import *
from .sim_op import *
import gzip
import json
import os

def dictToFile(dict, path):
    ''' Writes to gz format

	Args:
		dict `dict` file to save
		path `str` path to save it

	Returns:
		Nothing. Saves file to directory

    '''
    print("Writing to {}".format(path))
    with gzip.open(path, 'w') as f:
        f.write(json.dumps(dict))

def show_att(att):
    """ Show Att
    """
    print("==================================================")
    att = np.array(att)
    att = np.squeeze(att)
    print(att.shape)
    for i in range(len(att)):
        # print(att[i])
        # print(att[i].shape)
        ptr = np.argmax(att[i], axis=1)
        print(ptr)

def show_afm(afm):
    """ Show affinity matrix
    """
    print(afm)

def save_qual_data2(fp, mdl, batch, out_dict,index_word,
                    args=None):
    def sent_to_idx(sent):
        return [index_word[x] for x in sent]

    q1 = [sent_to_idx(x[0]) for x in batch]
    q2 = [sent_to_idx(x[2]) for x in batch]
    label = [x[-1] for x in batch]
    data = {
        'q1':q1,
        'q2':q2,
        'output':out_dict,
        'label':label,
        'args':vars(args)
    }

    if not os.path.exists(fp):
        os.makedirs(fp)
    dictToFile(data,fp + '/' + mdl)

def save_qual_data(fp, mdl, att1, att2, afm, afm2, batch, index_word,
                    args=None):
    """ Save qualitative data
    """
    def sent_to_idx(sent):
        return [index_word[x] for x in sent]

    q1 = [sent_to_idx(x[0]) for x in batch]
    q2 = [sent_to_idx(x[2]) for x in batch]
    # print(q1)
    # print(q1)
    label = [x[-1] for x in batch]
    # print(label)
    att1 = np.squeeze(np.array(att1)).tolist()
    att2 = np.squeeze(np.array(att2)).tolist()
    afm = np.array(afm).tolist()
    afm2 = np.array(afm2).tolist()
    data = {
        'q1':q1,
        'q2':q2,
        'att1':att1,
        'att2':att2,
        'afm':afm,
        'afm2':afm2,
        'label':label,
        'args':vars(args)
    }

    if not os.path.exists(fp):
        os.makedirs(fp)
    dictToFile(data,fp + '/' + mdl + '_' + str(args.num_heads))
