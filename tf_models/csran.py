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

def csran_model(self, q1_embed, q2_embed, q1_len, q2_len, q1_max,
                q2_max, force_model=None, score=1,
                reuse=None, features=None, extract_embed=False,
                side='', c1_embed=None, c2_embed=None, p1_embed=None,
                p2_embed=None, i1_embed=None, i2_embed=None, o1_embed=None,
                o2_embed=None, o1_len=None, o2_len=None, q1_mask=None,
                q2_mask=None):
    """ Co-Stack Residual Affinity Networks (CSRAN)
    Extends CAFE to stacked residual architecture
    """

    print("Learning Repr [{}]".format(side))
    print(q1_embed)
    print(q2_embed)
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

    print("Building CSRAN architecture..")
    bsz = tf.shape(q1_embed)[0]
    d = self.args.rnn_size
    q1_outs, q2_outs=[],[]
    q1, q2 = q1_embed, q2_embed
    self.rnn = cudnn_rnn

    for i in range(self.args.rnn_layers):
        with tf.variable_scope("encoding{}".format(i), reuse=reuse):
            if('CAFE' in self.args.rnn_type):
                q1, q2, ff = factor_flow2(q1, q2, q1_len, q2_len,
                        None, None,
                        mask_a=q1_mask, mask_b=q2_mask,
                        initializer=self.initializer,
                        factor=self.args.factor, factor2=32,
                        name='enccafe{}'.format(i),
                        att_type='SOFT')
            rnn = self.rnn(num_layers=1, num_units=d,
                        batch_size=bsz,
                        input_size=q1.get_shape().as_list()[-1],
                        keep_prob=self.args.rnn_dropout,
                        is_train=self.is_train,
                        rnn_type=self.args.rnn_type,
                        init=self.init)
            q1 = rnn(q1, seq_len=q1_len, var_drop=self.args.var_drop)
            q2 = rnn(q2, seq_len=q2_len, var_drop=self.args.var_drop)
            q1_outs.append(q1)
            q2_outs.append(q2)


    if('V2' in self.args.rnn_type):
        match_func = residual_alignment_v2
    else:
        match_func = residual_alignment
    if('NONE' in self.args.rnn_type):
        q1_aligned, q2_aligned, _, _, _ = co_attention(
                                    q1,
                                    q2,
                                    att_type="SOFT",
                                    pooling='MATRIX',
                                    mask_diag=False,
                                    kernel_initializer=self.init,
                                    activation=None,
                                    dropout=self.dropout,
                                    seq_lens=[q1_len, q2_len],
                                    transform_layers=0,
                                    name='coatt',
                                    reuse=reuse,
                                    mask_a=q1_mask,
                                    mask_b=q2_mask
                                        )
    else:
        q1 = tf.concat(q1_outs, 2)
        q2 = tf.concat(q2_outs, 2)
        if('INTRA' in self.args.rnn_type):
            q1_intra, _,_ = match_func(q1, q1, q1_mask, q2_mask, self.args.rnn_layers,
                                    name='intrares', init=self.init,
                                    is_train=self.is_train, dropout=self.dropout)
            q2_intra, _,_ = match_func(q2, q2, q2_mask, q2_mask, self.args.rnn_layers,
                                    name='intrares', init=self.init,
                                    is_train=self.is_train, dropout=self.dropout)
            q1 = tf.concat([q1, q1_intra], 2)
            q2 = tf.concat([q2, q2_intra], 2)
        q1_aligned, q2_aligned, _ = match_func(q1, q2, q1_mask, q2_mask,
                                self.args.rnn_layers,
                                name='resdot', init=self.init,
                                is_train=self.is_train, dropout=self.dropout)

    q2_output = tf.concat([q1_aligned, q2, q1_aligned - q2, q1_aligned * q2], 2)
    q1_output = tf.concat([q2_aligned, q1, q2_aligned - q1, q2_aligned * q1], 2)

    for j in range(self.args.aggr_layers):
        with tf.variable_scope("aggr{}".format(j), reuse=reuse):
            rnn = self.rnn(num_layers=1,
                        num_units=d,
                        batch_size=bsz,
                        input_size=q1_output.get_shape().as_list()[-1],
                        keep_prob=self.args.rnn_dropout,
                        is_train=self.is_train,
                        rnn_type=self.args.rnn_type,
                        init=self.init)
            q1_output = rnn(q1_output, seq_len=q1_len,
                            var_drop=self.args.var_drop)
            q2_output = rnn(q2_output, seq_len=q2_len,
                            var_drop=self.args.var_drop)
    q1_output = tf.reduce_sum(q1_output, 1)
    q2_output = tf.reduce_sum(q2_output, 1)

    try:
        # For summary statistics
        self.max_norm = tf.reduce_max(tf.norm(q1_output,
                                    ord='euclidean',
                                    keep_dims=True, axis=1))
    except:
        self.max_norm = 0

    if(extract_embed):
        self.q1_extract = q1_output
        self.q2_extract = q2_output

    # q1_output = tf.nn.dropout(q1_output, self.dropout)
    # q2_output = tf.nn.dropout(q2_output, self.dropout)

    if('SOFT' in self.args.rnn_type):
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


    elif('COSINE' in self.args.rnn_type):
        output = cosine_similarity(q1_output, q2_output)
    representation = output
    output = self._mlp_and_softmax(output, reuse=reuse, side=side)
    return output, representation, att1, att2
