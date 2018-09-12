#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
import gzip
import json
from tqdm import tqdm
import random
from collections import Counter
import operator
import timeit
import time

import datetime
from keras.preprocessing import sequence

from .utilities import *
from keras.utils import np_utils
import numpy as np

from tylib.lib.att_op import *
from tylib.lib.seq_op import *
from tylib.lib.rnn import *
from tylib.exp.tuning import *
from tylib.lib.cnn import *
from tylib.lib.compose_op import *
from tylib.lib.func import *

def cafe_model(self, q1_embed, q2_embed, q1_len, q2_len, q1_max,
            q2_max, force_model=None, score=1,
            reuse=None, features=None, extract_embed=False,
            side='', c1_embed=None, c2_embed=None, p1_embed=None,
            p2_embed=None, i1_embed=None, i2_embed=None, o1_embed=None,
            o2_embed=None, o1_len=None, o2_len=None, q1_mask=None,
            q2_mask=None):
    """ CAFE model

    q1_embed, q2_embed are premise/hypothesis respectively.
    q1_len, q2_len is a 1D tensor with their actual lengths
    q1_max, q2_max are maximum seq lengths for each sentence

    generally:
        1. c1,c2 are character embeddings
        2. p1, p2 are POS embeddings
        3. ignore i1, i2 (used in other experiments)
        4. ignore o1, o2. (used in other experiments)
    """

    print("Learning Repr [{}]".format(side))
    print(q1_embed)
    print(q2_embed)

    if(self.args.clip_embed>0):
        q1_embed = tf.clip_by_norm(q1_embed, self.args.clip_embed, axes=2)
        q2_embed = tf.clip_by_norm(q2_embed, self.args.clip_embed, axes=2)

    if('CHAR' in self.args.rnn_type):
        q1_embed = tf.concat([q1_embed, c1_embed], 2)
        q2_embed = tf.concat([q2_embed, c2_embed], 2)

    if(self.args.use_pos==1):
        q1_embed = tf.concat([q1_embed, p1_embed], 2)
        q2_embed = tf.concat([q2_embed, p2_embed], 2)

    translate_act = tf.nn.relu

    # Extra projection layer
    if('HP' in self.args.rnn_type):
        use_mode='HIGH'
    else:
        use_mode='FC'

    if(self.args.translate_proj==1):
        q1_embed = projection_layer(
                q1_embed,
                self.args.rnn_size,
                name='trans_proj',
                activation=translate_act,
                initializer=self.initializer,
                dropout=self.args.dropout,
                reuse=reuse,
                use_mode=use_mode,
                num_layers=self.args.num_proj,
                return_weights=True,
                is_train=self.is_train
                )
        q2_embed = projection_layer(
                q2_embed,
                self.args.rnn_size,
                name='trans_proj',
                activation=translate_act,
                initializer=self.initializer,
                dropout=self.args.dropout,
                reuse=True,
                use_mode=use_mode,
                num_layers=self.args.num_proj,
                is_train=self.is_train
                )
    else:
        self.proj_weights = self.embeddings

    if(self.args.all_dropout):
        q1_embed = tf.nn.dropout(q1_embed, self.dropout)
        q2_embed = tf.nn.dropout(q2_embed, self.dropout)

    representation = None

    att1, att2 = None, None
    if(force_model is not None):
        rnn_type = force_model
    else:
        rnn_type = self.args.rnn_type

    rnn_size = self.args.rnn_size
    if('INTER' in rnn_type):
        q1_embed, q2_embed, _, _, _ = co_attention(
                                    q1_embed,
                                    q2_embed,
                                    att_type=self.args.att_type,
                                    pooling='MATRIX',
                                    mask_diag=False,
                                    kernel_initializer=self.initializer,
                                    activation=None,
                                    dropout=self.args.dropout,
                                    seq_lens=[q1_len, q2_len],
                                    transform_layers=0,
                                    name='Ap_layer',
                                    reuse=reuse,
                                    is_train=self.is_train,
                                    mask_a=q1_mask,
                                    mask_b=q2_mask
                                        )
    elif("INTRA" in rnn_type):
        _q1_embed, _q2_embed = self._intra_attention(q1_embed, q2_embed,
                                            q1_len, q2_len,
                                            att_type=self.args.att_type,
                                            reuse=reuse,
                                            mask_a=q1_mask,
                                            mask_b=q2_mask
                                            )
        q1_embed = tf.concat([q1_embed, _q1_embed], 2)
        q2_embed = tf.concat([q2_embed, _q2_embed], 2)
    elif('FF' in rnn_type):
        q1_embed, q2_embed, fffeats = self.factor_flow2(
                                    q1_embed, q2_embed,
                                    q1_len, q2_len,
                                    q1_max, q2_max,
                                    reuse=reuse,
                                    name='word_ff',
                                    factor=self.args.factor,
                                    factor2=self.args.factor2,
                                    mask_a=q1_mask,
                                    mask_b=q2_mask)
        if(side=='POS'):
            self.ff_feats = fffeats

    q1_output = self.learn_single_repr(q1_embed, q1_len, q1_max,
                                    rnn_type,
                                    reuse=reuse, pool=False,
                                    name='main', mask=q1_mask)
    q2_output = self.learn_single_repr(q2_embed, q2_len, q2_max,
                                    rnn_type,
                                    reuse=True, pool=False,
                                    name='main', mask=q2_mask)

    print("==============================================")
    print('Representation Sizes:')
    print(q1_output)
    print(q2_output)
    print("===============================================")

    if('MEAN' in rnn_type):
        # Standard Mean Over Time Baseline
        q1_len = tf.expand_dims(q1_len, 1)
        q2_len = tf.expand_dims(q2_len, 1)
        q1_output = mean_over_time(q1_output, q1_len)
        q2_output = mean_over_time(q2_output, q2_len)
    elif('SUM' in rnn_type):
        q1_output = tf.reduce_sum(q1_output, 1)
        q2_output = tf.reduce_sum(q2_output, 1)
    elif('MAX' in rnn_type):
        q1_output = tf.reduce_max(q1_output, 1)
        q2_output = tf.reduce_max(q2_output, 1)
    elif('LAST' in rnn_type):
        q1_output = last_relevant(q1_output, q1_len)
        q2_output = last_relevant(q2_output, q2_len)
    elif('MM' in rnn_type):
        # max mean pooling
        q1_len = tf.expand_dims(q1_len, 1)
        q2_len = tf.expand_dims(q2_len, 1)
        q1_mean = mean_over_time(q1_output, q1_len)
        q2_mean = mean_over_time(q2_output, q2_len)
        q1_max = tf.reduce_max(q1_output, 1)
        q2_max = tf.reduce_max(q2_output, 1)
        q1_output = tf.concat([q1_mean, q1_max], 1)
        q2_output = tf.concat([q2_mean, q2_max], 1)

    try:
        # For summary statistics
        self.max_norm = tf.reduce_max(
                            tf.norm(q1_output,
                            ord='euclidean',
                            keep_dims=True, axis=1))
    except:
        self.max_norm = 0

    if(extract_embed):
        self.q1_extract = q1_output
        self.q2_extract = q2_output

    q1_output = tf.nn.dropout(q1_output, self.dropout)
    q2_output = tf.nn.dropout(q2_output, self.dropout)

    output = tf.concat([q1_output, q2_output], 1)

    if('SUB' in self.args.comp_layer):
        sub = q1_output - q2_output
        output = tf.concat([output, sub], 1)
    if('MUL' in self.args.comp_layer):
        comp = q1_output * q2_output
        output = tf.concat([output, comp], 1)
    if('L1' in self.args.comp_layer):
        l1_norm = tf.reduce_sum(tf.abs(q1_output - q2_output), 1,
                    keep_dims=True)
        output = tf.concat([output, l1_norm], 1)
    if('FM' in self.args.comp_layer):
        fm_feat, _ = build_fm(concat, k=self.args.factor,
                        name='final_fm',
                        initializer=self.initializer,
                        reshape=False, reuse=reuse)
        output = tf.concat([output, fm_feat], 1)

    penultimate = output
    output = self._mlp_and_softmax(output, reuse=reuse, side=side)
    representation = output

    return output, representation, att1, att2
