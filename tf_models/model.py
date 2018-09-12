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
import tensorflow_hub as hub

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
from tylib.lib.func import *
from tylib.lib.opt import *
from .cafe import *
from .csran import *

class Model:
    ''' TensorFlow Model for STS
    '''
    def __init__(self, vocab_size, args, char_vocab=0, pos_vocab=0,
                    mode='RANK', num_user=0, num_item=0):
        self.vocab_size = vocab_size
        self.char_vocab = char_vocab
        self.pos_vocab = pos_vocab
        self.graph = tf.Graph()
        self.args = args
        self.imap = {}
        self.inspect_op = []
        self.mode=mode
        self.write_dict = {}
        # For interaction data only
        self.num_user = num_user
        self.num_item = num_item
        print('Creating Model in [{}] mode'.format(self.mode))
        self.feat_prop = None
        if(self.args.init_type=='xavier'):
            self.initializer = tf.contrib.layers.xavier_initializer()
        elif(self.args.init_type=='normal'):
            self.initializer = tf.random_normal_initializer(0.0,
                                                    self.args.init)
        elif(self.args.init_type=='uniform'):
            self.initializer = tf.random_uniform_initializer(
                                                    maxval=self.args.init,
                                                    minval=-self.args.init)

        self.cnn_initializer = tf.random_uniform_initializer(
                                                maxval=self.args.init,
                                                minval=-self.args.init)
        self.init = self.initializer
        self.temp = []
        self.att1, self.att2 = [],[]
        self.build_graph()

    def _get_pair_feed_dict(self, data, mode='training', lr=None):
        # print(data[0])
        data = zip(*data)
        labels = data[-1]

        if(lr is None):
            lr = self.args.learn_rate
        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.q1_len:data[self.imap['q1_len']],
            self.q2_len:data[self.imap['q2_len']],
            self.learn_rate:lr,
            self.dropout:self.args.dropout,
            self.rnn_dropout:self.args.rnn_dropout,
            self.emb_dropout:self.args.emb_dropout
        }
        if(mode=='training'):
            feed_dict[self.q3_inputs] = data[self.imap['q3_inputs']]
            feed_dict[self.q3_len]=data[self.imap['q3_len']]
        if(self.args.use_elmo==1):
            feed_dict[self.q1_elmo_inputs] = data[self.imap['q1_elmo_inputs']]
            feed_dict[self.q2_elmo_inputs] = data[self.imap['q2_elmo_inputs']]
            if(mode=='training'):
                feed_dict[self.q3_elmo_inputs] = data[self.imap['q3_elmo_inputs']]
        if('CHAR' in self.args.rnn_type):
            feed_dict[self.c1_inputs] = data[self.imap['c1_inputs']]
            feed_dict[self.c2_inputs] = data[self.imap['c2_inputs']]
            if(mode=='training'):
                feed_dict[self.c3_inputs] = data[self.imap['c3_inputs']]
        if(mode!='training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.rnn_dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
        if(self.args.features):
            feed_dict[self.pos_features] = data[6]
            if(mode=='training'):
                feed_dict[self.neg_features] = data[7]
        return feed_dict

    def _check_model_type(self):
        if('SOFT' in self.args.rnn_type or 'EXP_MSE' in self.args.rnn_type):
            return 'point'
        elif('SIG_MSE' in self.args.rnn_type \
                or 'RAW_MSE' in self.args.rnn_type):
            return 'point'
        else:
            return 'pair'

    def get_feed_dict(self, data, mode='training', lr=None):
        mdl_type = self._check_model_type()
        if(mdl_type=='point'):
            return self._get_point_feed_dict(data, mode=mode, lr=lr)
        else:
            return self._get_pair_feed_dict(data, mode=mode, lr=lr)

    def _get_point_feed_dict(self, data, mode='training', lr=None):
        # print(data[0])
        data = zip(*data)

        labels = data[-1]
        if(self.args.dataset=='SICK'):
            soft_labels = encode_sick_labels(labels)
            sig_labels = encode_sick_labels_minmax(labels)
        else:
            soft_labels = np.array([[1 if t == i else 0
                                for i in range(self.args.num_class)] \
                                for t in labels])
            sig_labels = labels

        if(lr is None):
            lr = self.args.learn_rate
        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.q1_len:data[self.imap['q1_len']],
            self.q2_len:data[self.imap['q2_len']],
            self.learn_rate:lr,
            self.dropout:self.args.dropout,
            self.rnn_dropout:self.args.rnn_dropout,
            self.emb_dropout:self.args.emb_dropout,
            self.soft_labels:soft_labels,
            self.sig_labels:sig_labels
        }

        if(self.args.use_elmo==1):
            feed_dict[self.q1_elmo_inputs] = data[self.imap['q1_elmo_inputs']]
            feed_dict[self.q2_elmo_inputs] = data[self.imap['q2_elmo_inputs']]
        if('CHAR' in self.args.rnn_type):
            feed_dict[self.c1_inputs] = data[self.imap['c1_inputs']]
            feed_dict[self.c2_inputs] = data[self.imap['c2_inputs']]
        if(self.args.local_feats==1):
            feed_dict[self.f1_inputs] = data[self.imap['f1_inputs']]
            feed_dict[self.f2_inputs] = data[self.imap['f2_inputs']]
        if(self.args.use_pos==1):
            feed_dict[self.p1_inputs] = data[self.imap['p1_inputs']]
            feed_dict[self.p2_inputs] = data[self.imap['p2_inputs']]
        if(self.args.use_kg==1):
            feed_dict[self.k1_inputs] = data[self.imap['k1_inputs']]
            feed_dict[self.k2_inputs] = data[self.imap['k2_inputs']]

        if(mode!='training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.rnn_dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
        if(self.args.features):
            feed_dict[self.pos_features] = data[6]
        return feed_dict

    def register_index_map(self, idx, target):
        self.imap[target] = idx

    def build_rnn(self, embed, lengths, max_len,
                    reuse=None, force_model=None,
                    initializer=None, name=''):
        print("Build RNN={}".format(reuse))
        if(self.args.rnn_constant==1):
            _dim = embed.get_shape().as_list()[2]
            print("Setting RNN to constant size as input dim={}".format(_dim))
            rnn_size = _dim
        else:
            rnn_size = self.args.rnn_size
        if(force_model is not None):
            rnn_type = force_model
        else:
            rnn_type = self.args.rnn_type
        with tf.variable_scope('RNN_{}'.format(name),
                        initializer=initializer, reuse=reuse) as scope:
            if('LSTM' in rnn_type):
                scell = tf.contrib.rnn.BasicLSTMCell(
                                        rnn_size,
                                        forget_bias=1.0,
                                        state_is_tuple=True,
                                        reuse=reuse)
            elif('GRU' in rnn_type):
                scell = tf.contrib.rnn.GRUCell(
                                        rnn_size,
                                        reuse=reuse)
            if(self.args.dropout<1.0):
                scell = tf.contrib.rnn.DropoutWrapper(scell,
                                        output_keep_prob=self.rnn_dropout)
            if(self.args.rnn_layers>1):
                scell = tf.contrib.rnn.MultiRNNCell(
                        [tf.contrib.rnn.BasicLSTMCell(
                        self.args.rnn_size, forget_bias=1.0, reuse=reuse)
                         for _ in range(self.args.rnn_layers)],
                         state_is_tuple=True)
            init_state = scell.zero_state(tf.shape(embed)[0], tf.float32)
            if('BI' in rnn_type):
                backward = tf.contrib.rnn.BasicLSTMCell(rnn_size,
                                                    forget_bias=1.0,
                                                    state_is_tuple=True,
                                                    reuse=reuse)
                sentence_outputs, s_last = tf.nn.bidirectional_dynamic_rnn(\
                                                cell_fw=scell,\
                                                cell_bw=backward,\
                                                inputs=embed,\
                                                dtype=tf.float32,\
                                                initial_state_fw=init_state,
                                                sequence_length=tf.cast(
                                                        lengths, tf.int32)\
                                                                )
                sentence_outputs = tf.concat(sentence_outputs, 2)
            else:
                sentence_outputs, s_last = tf.nn.dynamic_rnn(scell, embed,
                                                sequence_length=tf.cast(
                                                            lengths,
                                                             tf.int32),
                                                initial_state=init_state,
                                                dtype=tf.float32)
        # sentence_outputs = mask_zeros_1(sentence_outputs, lengths, max_len)
        return sentence_outputs

    def build_glove(self, embed, lens, max_len):
        # embed = mask_zeros_1(embed, lens, max_len)
        return tf.reduce_sum(embed, 1)

    def _intra_attention(self, q1_embed, q2_embed, q1_len, q2_len,
                        att_type='SOFT', reuse=None, proj=False,
                        name="", dist_bias=0,
                        mask_a=None, mask_b=None):
        """ Computes intra_attention
        """
        _q1_embed, _, _, _, _ = co_attention(
                                    q1_embed,
                                    q1_embed,
                                    att_type=att_type,
                                    pooling='MATRIX',
                                    mask_diag=False,
                                    kernel_initializer=self.initializer,
                                    activation=None,
                                    dropout=self.args.dropout,
                                    seq_lens=[q1_len, q1_len],
                                    transform_layers=self.args.num_intra_proj,
                                    name='{}_intra_layer'.format(name),
                                    reuse=reuse,
                                    dist_bias=dist_bias,
                                    mask_a=mask_a,
                                    mask_b=mask_a
                                    )
        _q2_embed, _, _, _, _ = co_attention(
                                q2_embed,
                                q2_embed,
                                att_type=att_type,
                                pooling='MATRIX',
                                mask_diag=False,
                                kernel_initializer=self.initializer,
                                activation=None,
                                dropout=self.args.dropout,
                                seq_lens=[q2_len, q2_len],
                                transform_layers=self.args.num_intra_proj,
                                name='{}_intra_layer'.format(name),
                                reuse=True,
                                dist_bias=dist_bias,
                                mask_a=mask_b,
                                mask_b=mask_b
                                )

        if(self.args.proj_intra==1):
            q1_embed = tf.concat([q1_embed, _q1_embed], 2)
            q2_embed = tf.concat([q2_embed, _q2_embed], 2)
            q1_embed = projection_layer(q1_embed,
                        self.args.rnn_size,
                        name='{}_intra_proj'.format(name),
                        activation=tf.nn.relu,
                        initializer=self.initializer,
                        dropout=self.args.dropout,
                        reuse=reuse,
                        num_layers=1,
                        is_train=self.is_train)

            q2_embed = projection_layer(q2_embed,
                        self.args.rnn_size,
                        name='{}_intra_proj'.format(name),
                        activation=tf.nn.relu,
                        initializer=self.initializer,
                        dropout=self.args.dropout,
                        reuse=True,
                        num_layers=1,
                        is_train=self.is_train)
        return  _q1_embed, _q2_embed

    def feat_compare(self, q1_compare, q2_compare, name='', reuse=None):
        """ FM layers for feature comparison
        """
        q1_fm, q1_latent = build_fm(q1_compare, k=self.args.factor,
                            name='{}_com_fm'.format(name),
                            initializer=self.initializer,
                            reshape=True, reuse=reuse)
        q2_fm, q2_latent = build_fm(q2_compare, k=self.args.factor,
                            name='{}_com_fm'.format(name),
                            initializer=self.initializer, reuse=True,
                            reshape=True)
        return q1_fm, q2_fm, q1_fm, q2_fm

    def alignment_compare(self, q1, q2, _q1, _q2, reuse=None,
                        feature_list=[], name=''):
        """ Compares and build features between alignments

        q1 will compare with _q2
        q2 will compare with _q1
        """

        features1 = []
        features2 = []
        latent_features1 = []
        latent_features2 = []

        for mode in feature_list:
            mname = name + mode
            if(mode=='CAT'):
                fv1 = tf.concat([q1, _q2], 2)
                fv2 = tf.concat([q2, _q1], 2)
                f1, f2, l1, l2 = self.feat_compare(fv1, fv2, name=mname,
                                                    reuse=reuse)
            elif(mode=='MUL'):
                fv1 = q1 * _q2
                fv2 = q2 * _q1
                f1, f2, l1, l2 = self.feat_compare(fv1, fv2, name=mname,
                                                    reuse=reuse)
            elif(mode=='SUB'):
                fv1 = q1 - _q2
                fv2 = q2 - _q1
                f1, f2, l1, l2 = self.feat_compare(fv1, fv2, name=mname,
                                                    reuse=reuse)
            features1.append(f1)
            features2.append(f2)
            latent_features1.append(l1)
            latent_features2.append(l2)
        return features1, features2, latent_features1, latent_features2

    def coatt_feats(self, q1_embed, q2_embed, q1_len, q2_len,
                    q1_max, q2_max,
                    reuse=None, name='', pooling='MAX'):

        features1 = []
        features2 = []
        la_feats1 = []
        la_feats2 = []

        # By convention we should pass it in like MUL_SUB_CAT
        feature_list = self.args.fm_comp_layer.split('_')

        _q1_embed, _q2_embed, _, _, afm = co_attention(
                            q1_embed,
                            q2_embed,
                            att_type=self.args.att_type,
                            pooling=pooling,
                            mask_diag=False,
                            kernel_initializer=self.initializer,
                            activation=None,
                            dropout=self.args.dropout,
                            seq_lens=[q1_len, q2_len],
                            transform_layers=self.args.num_inter_proj,
                            name='{}_ca_layer'.format(name),
                            reuse=reuse,
                            is_train=self.is_train
                            )
        self.write_dict['afm_{}'.format(pooling)] = afm

        f1, f2, l1, l2 = self.alignment_compare(q1_embed, q2_embed,
                             _q2_embed, _q1_embed,
                            reuse=reuse,
                            feature_list=feature_list,
                            name='{}_inter_{}'.format(name, pooling))
        features1 += f1
        features2 += f2
        la_feats1 += l1
        la_feats2 += l2

        features1 = tf.concat(features1, 2)
        features2 = tf.concat(features2, 2)

        return features1, features2

    def factor_flow2(self, q1_embed, q2_embed, q1_len, q2_len,
                    q1_max, q2_max,
                    factor=16, factor2=4,
                    reuse=None, name='',
                    mask_a=None, mask_b=None):
        """ Improved Factor Flow for NLI data
        """

        if('RFF' in self.args.rnn_type):
            # Recurrent model
            print("Adding PreRNN")
            with tf.variable_scope('pre_RNN',
                            initializer=self.initializer) as scope:
                    q1_embed = self.build_rnn(q1_embed, q1_len,
                                        q1_max, reuse=reuse)
            with tf.variable_scope('pre_RNN',
                                initializer=self.initializer) as scope:
                    q2_embed = self.build_rnn(q2_embed, q2_len,
                                    q2_max, reuse=True)

        features1 = []
        features2 = []
        la_feats1 = []
        la_feats2 = []

        # By convention we should pass it in like MUL_SUB_CAT
        feature_list = self.args.fm_comp_layer.split('_')

        if(self.args.enc_only==0 and 'ALIGN' in self.args.mca_types):
            # ___ Inter Attention Features
            _q1_embed, _q2_embed, _, _, afm = co_attention(
                                q1_embed,
                                q2_embed,
                                att_type=self.args.att_type,
                                pooling='MATRIX',
                                mask_diag=False,
                                kernel_initializer=self.initializer,
                                activation=None,
                                dropout=self.args.dropout,
                                seq_lens=[q1_len, q2_len],
                                transform_layers=self.args.num_inter_proj,
                                name='{}_ca_layer'.format(name),
                                reuse=reuse,
                                mask_a=mask_a,
                                mask_b=mask_b
                                )
            self.write_dict['afm_align'] = afm

            f1, f2, l1, l2 = self.alignment_compare(q1_embed, q2_embed,
                                 _q1_embed, _q2_embed,
                                reuse=reuse,
                                feature_list=feature_list,
                                name='{}_inter'.format(name))
            features1 += f1
            features2 += f2
            la_feats1 += l1
            la_feats2 += l2

        # ___ Intra Attention

        if('INTRA' in self.args.mca_types):
            _i1_embed, _i2_embed =  self._intra_attention(q1_embed, q2_embed,
                                            q1_len, q2_len,
                                            att_type='SOFT',
                                            name=name,
                                            reuse=reuse,
                                            dist_bias=self.args.dist_bias,
                                            mask_a=mask_a, mask_b=mask_b)

            f1, f2, l1, l2 = self.alignment_compare(q1_embed, q2_embed,
                                     _i2_embed, _i1_embed,
                                    reuse=reuse,
                                    feature_list=feature_list,
                                    name='{}_intra'.format(name))
            features1 += f1
            features2 += f2
            la_feats1 += l1
            la_feats2 += l2

        features1 = tf.concat(features1, 2)
        features2 = tf.concat(features2, 2)

        q2_embed = tf.concat([q2_embed, features2], 2)
        q1_embed = tf.concat([q1_embed, features1], 2)

        if(self.args.flow_meta==1 and self.args.enc_only==0):
            print("Using Meta Flow")
            # Should do something about this?
            # Is this really helpful?
            la_feats1 = tf.concat(la_feats1, 2)
            la_feats2 = tf.concat(la_feats2, 2)
            f1, f1l = build_fm(la_feats1, k=factor2,
                                name='{}_aggregate_fm'.format(name),
                                initializer=self.initializer,
                                reshape=True, reuse=reuse)
            f2, f2l = build_fm(la_feats2, k=factor2,
                                name='{}_aggregate_fm'.format(name),
                                initializer=self.initializer,
                                reshape=True, reuse=True)
            q2_embed = tf.concat([q2_embed, f2], 2)
            q1_embed = tf.concat([q1_embed, f1], 2)
        if(self.args.flow_intra==1 and self.args.enc_only==0):
            print("Adding IntraAtt to Repr")
            q2_embed = tf.concat([q2_embed, _i2_embed], 2)
            q1_embed = tf.concat([q1_embed, _i1_embed],2)
        if(self.args.flow_inner==1):
            print("Adding InnerAtt to Repr")
            q2_embed = tf.concat([q2_embed, _q1_embed], 2)
            q1_embed = tf.concat([q1_embed, _q2_embed],2)
        print("====================================================")
        print("Factor Flow Embedding")
        print(q1_embed)
        print(q2_embed)
        print("====================================================")
        return q1_embed, q2_embed, [features1, features2]

    def learn_single_repr(self, q1_embed, q1_len, q1_max, rnn_type,
                        reuse=None, pool=False, name="", mask=None):
        print(q1_len)
        print(rnn_type)
        rnn_selected = False
        if('GLOVE' in rnn_type):
            # print(q1_len)
            # q1_output = self.build_glove(q1_embed, q1_len, q1_max)
            if('ATT' in rnn_type):
                q1_output, att = attention(q1_embed,
                                context=None, reuse=reuse, name='',
                                kernel_initializer=self.initializer,
                                dropout=None)
            else:
                q1_output = tf.reduce_sum(q1_embed, 1)
            if(pool):
                return q1_embed, q1_output
        elif('CNN' in rnn_type):
            q1_output = build_raw_cnn(q1_embed, self.args.rnn_size,
                filter_sizes=3,
                initializer=self.initializer,
                dropout=self.rnn_dropout, reuse=reuse, name=name)
            if(pool):
                q1_output = tf.reduce_max(q1_output, 1)
                return q1_output, q1_output
        elif('SELF-ATT' in rnn_type):
            _q1, _, _, _, _ = co_attention(
                    q1_embed, q1_embed,
                    att_type='BILINEAR',
                    pooling='MATRIX',
                    mask_diag=True,
                    kernel_initializer=self.initializer,
                    activation=None,
                    dropout=self.dropout,
                    seq_lens=None,
                    transform_layers=1,
                    proj_activation=None,
                    name='selfenc',
                    reuse=reuse,
                    mask_a=mask, mask_b=mask
                    )
            q1_output = tf.concat([q1_embed, _q1], 2)
            new_q1_output = projection_layer(q1_output,
                        self.args.rnn_size,
                        name='selfatt_proj',
                        activation=None,
                        initializer=self.initializer,
                        dropout=self.args.dropout,
                        reuse=reuse,
                        num_layers=1,
                        mode='FC',
                        is_train=self.is_train)

            if('GATE' in rnn_type):
                gate = tf.nn.sigmoid(new_q1_output)
                q1_output = (gate * _q1) + (1-gate) * q1_embed
            else:
                q1_output = new_q1_output
            q1_output = tf.reduce_sum(q1_output, 1)
            return q1_output, q1_output
        elif('RNN' in rnn_type or 'LSTM' in rnn_type \
                or 'GRU' in rnn_type or 'RORU' in rnn_type):
            # LSTM, GRU and RNN
            if(self.args.rnn_init_type=='orth'):
                oinit = tf.orthogonal_initializer()
            else:
                oinit = self.initializer

            q1_output, _ = build_rnn(q1_embed, q1_len,
                            rnn_type=self.args.rnn_type,
                            reuse=reuse,
                            rnn_dim=self.args.rnn_size,
                            dropout=self.args.dropout,
                            initializer=oinit,
                            use_cudnn=self.args.use_cudnn,
                            is_train=self.is_train,
                            train_init=self.args.train_rnn_init)
            # q1_output = mask_zeros_1(q1_output, q1_len, q1_max)
            rnn_selected = True
            if(pool==True):
                if('MEAN' in rnn_type):
                    # Standard Mean Over Time Baseline
                    vec = mean_over_time(q1_output,
                                        tf.expand_dims(q1_len, 1))
                elif('LAST' in rnn_type):
                    # Gets the Last LSTM output
                    vec = last_relevant(q1_output, q1_len)
                elif('ATT' in rnn_type):
                    vec, att = attention(q1_output,
                                    context=None, reuse=reuse, name='',
                                    kernel_initializer=self.initializer,
                                    dropout=None)
                elif('MAX' in rnn_type):
                    vec = tf.reduce_max(q1_output, 1)
                else:
                    vec = None
                return q1_output,  vec
        else:
            q1_output = q1_embed

        return q1_output

    def _mlp_and_softmax(self, output, reuse=None, side=''):
        """ builds MLP and softmax (prediction layers of the network)
        """
        with tf.variable_scope('fully_connected', reuse=reuse) as scope:
            if(self.args.hdim>0 and self.args.num_dense>0):
                for i in range(0, self.args.num_dense):
                    if('HIGH' in self.args.rnn_type):
                        # use highway instead
                        output = highway_layer(output, self.args.hdim,
                                self.initializer,
                                name='high_{}'.format(i), reuse=reuse,
                                activation=tf.nn.relu)
                    else:
                        output = ffn(output, self.args.hdim, self.initializer,
                                    name='fc_{}'.format(i),
                                    reuse=None, activation=tf.nn.relu )
                    output = tf.nn.dropout(output, self.dropout)

            if(self.args.final_layer==1):
                # This layer should be set to one
                # except for cosine ranking
                activation = None
                with tf.variable_scope('fl',reuse=reuse) as scope:
                    if('SOFT' in self.args.rnn_type):
                        num_outputs = self.args.num_class
                    else:
                        num_outputs = 1

                    last_dim = output.get_shape().as_list()[1]
                    weights_linear = tf.get_variable('final_weights',
                                    [last_dim, num_outputs],
                                        initializer=self.initializer)
                    bias_linear = tf.get_variable('bias',
                                [num_outputs],
                                    initializer=tf.zeros_initializer())

                    if('NB' in self.args.rnn_type):
                        # no bias (not really used)
                        final_layer = tf.matmul(output, weights_linear)
                    else:
                        final_layer = tf.nn.xw_plus_b(output, weights_linear,
                                                    bias_linear)
                    output = final_layer
                    if('SIGR' in self.args.rnn_type):
                        # Sigmoid output (not used here)
                        output = tf.nn.sigmoid(output)
        return output

    def _char_cnn(self, cin, max_len, output_dim, clens,
                                reuse=None):
        print("============================")
        print("Char CNN Encoder..")
        _char_len = tf.reshape(cin, [-1, self.args.char_max])
        _char_len = tf.cast(tf.cast(_char_len, tf.bool), tf.int32)
        _char_len = tf.reduce_sum(_char_len, 1)
        ce = tf.nn.embedding_lookup(self.char_embed,
                                    cin)
        ce = tf.reshape(ce, [-1, self.args.char_max,
                                self.args.char_emb_size])
        if(self.args.char_enc=='CNN'):
            _cnn = build_cnn(ce, output_dim,
                            filter_sizes=self.args.conv_size,
                            initializer=self.initializer,
                            reuse=reuse)
            _final_output = output_dim
        elif(self.args.char_enc=='SUM'):
            _cnn = tf.reduce_sum(ce, 1)
            _final_output = self.args.char_emb_size
        elif(self.args.char_enc=='RNN'):
            bsz = tf.shape(ce)[0]
            with tf.variable_scope('charRNN', reuse=reuse) as scope:
                char_rnn = cudnn_rnn(
                            num_layers=1,
                            num_units=self.args.cnn_size,
                            batch_size=bsz,
                            input_size=self.args.char_emb_size,
                            keep_prob=self.args.dropout,
                            is_train=self.is_train,
                            rnn_type='LSTM')
                _cnn = char_rnn(ce, seq_len=_char_len,
                                        var_drop=self.args.var_drop)
                _cnn = last_relevant(_cnn, _char_len)
                _final_output = output_dim * 2

        # out_cnn = tf.reshape(_cnn, [-1, max_len])
        out_cnn = tf.reshape(_cnn, [-1, max_len])
        if(self.args.clip_sent==1):
            out_cnn, cmax = clip_sentence(out_cnn, clens)
        else:
            cmax = max_len
        out_cnn = tf.reshape(out_cnn, [-1, cmax, _final_output])
        return out_cnn

    def _local_feats(self, fin, max_len, output_dim, flens):
        fin = tf.reshape(fin, [-1, max_len])
        fin, fmax = clip_sentence(fin, flens)
        out_fin = tf.reshape(fin, [-1, fmax, output_dim])
        return out_fin

    def prepare_inputs(self):
        """ Prepares Input
        """
        if(self.args.clip_sent==1):
            q1_inputs, self.qmax = clip_sentence(self.q1_inputs, self.q1_len)
            q2_inputs, self.a1max = clip_sentence(self.q2_inputs, self.q2_len)
            q3_inputs, self.a2max = clip_sentence(self.q3_inputs, self.q3_len)
        else:
            q1_inputs = self.q1_inputs
            q2_inputs = self.q2_inputs
            q3_inputs = self.q3_inputs
            self.qmax = tf.reduce_max(self.q1_len)
            self.a1max = tf.reduce_max(self.q2_len)
            self.a2max = tf.reduce_max(self.q3_len)

        self.q1_mask = tf.cast(q1_inputs, tf.bool)
        self.q2_mask = tf.cast(q2_inputs, tf.bool)
        self.q3_mask = tf.cast(q3_inputs, tf.bool)

        f1_inputs = self._local_feats(self.f1_inputs, self.args.qmax,
                                    1, self.q1_len)
        f2_inputs = self._local_feats(self.f2_inputs, self.args.amax,
                                    1, self.q2_len)
        f3_inputs = self._local_feats(self.f3_inputs, self.args.amax,
                                    1, self.q3_len)

        k1_inputs = self._local_feats(self.k1_inputs, self.args.qmax,
                                    3, self.q1_len)
        k2_inputs = self._local_feats(self.k2_inputs, self.args.amax,
                                    3, self.q2_len)
        k3_inputs = self._local_feats(self.k3_inputs, self.args.amax,
                                    3, self.q3_len)

        with tf.device('/cpu:0'):
            q1_embed =  tf.nn.embedding_lookup(self.embeddings,
                                                    q1_inputs)
            q2_embed =  tf.nn.embedding_lookup(self.embeddings,
                                                    q2_inputs)
            q3_embed = tf.nn.embedding_lookup(self.embeddings,
                                                    q3_inputs)

        if(self.args.all_dropout):
            q1_embed = tf.nn.dropout(q1_embed, self.emb_dropout)
            q2_embed = tf.nn.dropout(q2_embed, self.emb_dropout)
            q3_embed = tf.nn.dropout(q3_embed, self.emb_dropout)

        if(self.args.local_feats==1):
            print("Adding Local Features..")
            q1_embed = tf.concat([q1_embed, f1_inputs], 2)
            q2_embed = tf.concat([q2_embed, f2_inputs], 2)
            q3_embed = tf.concat([q3_embed, f3_inputs], 2)

        if(self.args.use_kg==1):
            print("Adding KG features..")
            q1_embed = tf.concat([q1_embed, k1_inputs], 2)
            q2_embed = tf.concat([q2_embed, k2_inputs], 2)
            q3_embed = tf.concat([q3_embed, k3_inputs], 2)

        if(self.args.use_cove==1):
            print("Adding Cove Embeddings")
            def add_cove(embed, cove_lstm):
                cove_embed = cove_lstm(embed)
                embed = tf.concat([embed, cove_embed], 2)
                return embed
            cove_lstm = load_cudnn_cove_lstm()
            q1_embed = add_cove(q1_embed, cove_lstm)
            q2_embed = add_cove(q2_embed, cove_lstm)
            q3_embed = add_cove(q3_embed, cove_lstm)

        if(self.args.use_elmo==1):
            print("Adding ELMO embeddings :p")
            self.elmo_model = hub.Module("https://tfhub.dev/google/elmo/2",
                                    trainable=True)
            def get_elmo(x, lengths):
                elmo = self.elmo_model(
                                inputs={
                                    "tokens": x,
                                    "sequence_len": lengths
                                  },
                                signature="tokens",
                                as_dict=True)["elmo"]
                elmo = tf.nn.dropout(elmo, self.emb_dropout)
                return elmo
            if(self.args.clip_sent==1):
                q1_elmo, _ = clip_sentence(self.q1_elmo_inputs, self.q1_len)
                q2_elmo, _ = clip_sentence(self.q2_elmo_inputs, self.q2_len)
                q3_elmo, _ = clip_sentence(self.q3_elmo_inputs, self.q3_len)
            else:
                q1_elmo = self.q1_elmo_inputs
                q2_elmo = self.q2_elmo_inputs
                q3_elmo = self.q3_elmo_inputs
            q1_elmo = get_elmo(q1_elmo, self.q1_len)
            q2_elmo = get_elmo(q2_elmo, self.q2_len)
            q3_elmo = get_elmo(q3_elmo, self.q3_len)
            q1_embed = tf.concat([q1_embed, q1_elmo], 2)
            q2_embed = tf.concat([q2_embed, q2_elmo], 2)
            q3_embed = tf.concat([q3_embed, q3_elmo], 2)

        if('CHAR' in self.args.rnn_type):
            self.char_embed = tf.get_variable('char_embed',
                                    [self.char_vocab,
                                    self.args.char_emb_size],
                                    initializer=self.initializer)
            self.c1_cnn = self._char_cnn(self.c1_inputs, self.args.qmax,
                            self.args.cnn_size, self.q1_len)
            self.c2_cnn = self._char_cnn(self.c2_inputs, self.args.amax,
                            self.args.cnn_size, self.q2_len,
                                reuse=True)
            self.c3_cnn = self._char_cnn(self.c3_inputs, self.args.amax,
                            self.args.cnn_size,
                            self.q3_len, reuse=True)
        else:
            self.c1_cnn, self.c2_cnn, self.c3_cnn = None, None, None

        if(self.args.use_pos==1):
            self.pos_embed = tf.get_variable('pos_embed',
                                    [self.pos_vocab,
                                    self.args.pos_emb_size],
                                    initializer=self.initializer)
            if(self.args.clip_sent==1):
                p1_inputs, _ = clip_sentence(self.p1_inputs, self.q1_len)
                p2_inputs, _ = clip_sentence(self.p2_inputs, self.q2_len)
                p3_inputs, _= clip_sentence(self.p3_inputs, self.q3_len)
            else:
                p1_inputs = self.p1_inputs
                p2_inputs = self.p2_inputs
                p3_inputs = self.p3_inputs
            self.p1_pos = tf.nn.embedding_lookup(self.pos_embed, p1_inputs)
            self.p2_pos = tf.nn.embedding_lookup(self.pos_embed, p2_inputs)
            self.p3_pos = tf.nn.embedding_lookup(self.pos_embed, p3_inputs)
        else:
            self.p1_pos, self.p2_pos, self.p3_pos = None, None, None

        self.q1_embed = q1_embed
        self.q2_embed = q2_embed
        self.q3_embed = q3_embed

    def build_graph(self):
        ''' Builds Computational Graph
        '''
        if(self.mode=='HREC' and self.args.base_encoder!='Flat'):
            len_shape = [None, None]
        else:
            len_shape = [None]

        print("Building placeholders with shape={}".format(len_shape))


        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False)
            self.is_train = tf.get_variable("is_train",
                                            shape=[],
                                            dtype=tf.bool,
                                            trainable=False)
            self.true = tf.constant(True, dtype=tf.bool)
            self.false = tf.constant(False, dtype=tf.bool)
            with tf.name_scope('q1_input'):
                self.q1_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.qmax],
                                                    name='q1_inputs')
                self.c1_inputs = tf.placeholder(tf.int32, shape=[None,
                                    self.args.char_max * self.args.qmax],
                                                    name='c1_inputs')
                self.f1_inputs = tf.placeholder(tf.float32, shape=[None,
                                                    self.args.qmax, 1],
                                                    name='f1_inputs')
                self.k1_inputs = tf.placeholder(tf.float32, shape=[None,
                                                    self.args.qmax, 3],
                                                    name='k1_inputs')
                self.p1_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.qmax],
                                                    name='p1_inputs')
                self.q1_elmo_inputs = tf.placeholder(tf.string, shape=[None,
                                                    self.args.qmax],
                                                    name='q1_elmo_inputs')

            with tf.name_scope('q2_input'):
                self.q2_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.amax],
                                                    name='q2_inputs')
                self.c2_inputs = tf.placeholder(tf.int32, shape=[None,
                                    self.args.char_max * self.args.amax],
                                                    name='c2_inputs')
                self.f2_inputs = tf.placeholder(tf.float32, shape=[None,
                                                    self.args.amax, 1],
                                                    name='f2_inputs')
                self.k2_inputs = tf.placeholder(tf.float32, shape=[None,
                                                    self.args.amax, 3],
                                                    name='f2_inputs')
                self.p2_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.amax],
                                                    name='p2_inputs')
                self.q2_elmo_inputs = tf.placeholder(tf.string, shape=[None,
                                                    self.args.amax],
                                                    name='q2_elmo_inputs')

            with tf.name_scope('q3_input'):
                self.q3_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.amax],
                                                    name='q3_inputs')
                self.c3_inputs = tf.placeholder(tf.int32, shape=[None,
                                    self.args.char_max * self.args.amax],
                                                    name='c3_inputs')
                self.f3_inputs = tf.placeholder(tf.float32, shape=[None,
                                                    self.args.amax, 1],
                                                    name='f3_inputs')
                self.k3_inputs = tf.placeholder(tf.float32, shape=[None,
                                                    self.args.amax, 3],
                                                    name='f3_inputs')
                self.p3_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.amax],
                                                    name='p3_inputs')
                self.q3_elmo_inputs = tf.placeholder(tf.string, shape=[None,
                                                    self.args.amax],
                                                    name='q3_elmo_inputs')

            with tf.name_scope('dropout'):
                self.dropout = tf.placeholder(tf.float32,
                                                name='dropout')
                self.rnn_dropout = tf.placeholder(tf.float32,
                                                name='rnn_dropout')
                self.emb_dropout = tf.placeholder(tf.float32,
                                                name='emb_dropout')
            with tf.name_scope('q1_lengths'):
                self.q1_len = tf.placeholder(tf.int32, shape=len_shape)
                self.c1_len = tf.placeholder(tf.int32, shape=[None, None])
            with tf.name_scope('q2_lengths'):
                self.q2_len = tf.placeholder(tf.int32, shape=len_shape)
                self.c2_len = tf.placeholder(tf.int32, shape=[None, None])
            with tf.name_scope('q3_lengths'):
                self.q3_len = tf.placeholder(tf.int32, shape=len_shape)
                self.c3_len = tf.placeholder(tf.int32, shape=[None, None])
            with tf.name_scope('learn_rate'):
                self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')
            if(self.args.pretrained==1):
                self.emb_placeholder = tf.placeholder(tf.float32,
                            [self.vocab_size, self.args.emb_size])
            with tf.name_scope('pos_features'):
                self.pos_features = tf.placeholder(tf.float32, shape=[None,
                                    self.args.num_extra_features])
            with tf.name_scope('neg_features'):
                self.neg_features = tf.placeholder(tf.float32, shape=[None,
                                    self.args.num_extra_features])

            with tf.name_scope("soft_labels"):
                # softmax cross entropy
                if(self.args.dataset=='SICK'):
                    data_type = tf.float32
                else:
                    data_type = tf.int32
                self.soft_labels = tf.placeholder(data_type,
                             shape=[None, self.args.num_class],
                             name='softmax_labels')

            with tf.name_scope("sig_labels"):
                # sigmoid cross entropy
                self.sig_labels = tf.placeholder(tf.float32,
                                                shape=[None],
                                                name='sigmoid_labels')
                self.sig_target = tf.expand_dims(self.sig_labels, 1)

            self.batch_size = tf.shape(self.q1_inputs)[0]

            with tf.variable_scope('embedding_layer'):
                if(self.args.pretrained==1):
                    self.embeddings = tf.Variable(tf.constant(
                                        0.0, shape=[self.vocab_size,
                                            self.args.emb_size]), \
                                        trainable=self.args.trainable,
                                         name="embeddings")
                    self.embeddings_init = self.embeddings.assign(
                                        self.emb_placeholder)
                else:
                    self.embeddings = tf.get_variable('embedding',
                                        [self.vocab_size,
                                        self.args.emb_size],
                                        initializer=self.initializer)
            # This is not used, please ignore i1,i2,i3
            self.i1_embed, self.i2_embed, self.i3_embed = None, None, None
            self.prepare_inputs()
            q1_len = self.q1_len
            q2_len = self.q2_len
            q3_len = self.q3_len
            self.o1_embed = None
            self.o2_embed = None
            self.o3_embed = None
            self.o1_len = None
            self.o2_len = None
            self.o3_len = None

            if('RESFM' in self.args.rnn_type):
                model_func = csran_model
            else:
                model_func = cafe_model

            self.output_pos, _, _, _ = model_func(self,
                                        self.q1_embed, self.q2_embed,
                                        q1_len, q2_len,
                                        self.qmax,
                                        self.a1max, score=1, reuse=None,
                                        features=self.pos_features,
                                        extract_embed=True,
                                        side='POS',
                                        c1_embed=self.c1_cnn,
                                        c2_embed=self.c2_cnn,
                                        p1_embed=self.p1_pos,
                                        p2_embed=self.p2_pos,
                                        i1_embed=self.i1_embed,
                                        i2_embed=self.i2_embed,
                                        o1_embed=self.o1_embed,
                                        o2_embed=self.o2_embed,
                                        o1_len=self.o1_len,
                                        o2_len=self.o2_len,
                                        q1_mask=self.q1_mask,
                                        q2_mask=self.q2_mask
                                        )
            if('SOFT' not in self.args.rnn_type and 'RAW_MSE' not in self.args.rnn_type):
                print("Building Negative Graph...")
                # Not used here for NLI
                self.output_neg,_,_, _ = model_func(self, self.q1_embed,
                                             self.q3_embed, q1_len,
                                             q3_len, self.qmax,
                                             self.a2max, score=1,
                                             reuse=True,
                                             features=self.neg_features,
                                             side='NEG',
                                             c1_embed=self.c1_cnn,
                                             c2_embed=self.c3_cnn,
                                             p1_embed=self.p1_pos,
                                             p2_embed=self.p3_pos,
                                             i1_embed=self.i1_embed,
                                             i2_embed=self.i3_embed,
                                             o1_embed=self.o1_embed,
                                             o2_embed=self.o3_embed,
                                             o1_len=self.o1_len,
                                             o2_len=self.o3_len,
                                             q1_mask=self.q1_mask,
                                             q2_mask=self.q3_mask
                                             )
            else:
                self.output_neg = None

            # Define loss and optimizer
            with tf.name_scope("train"):
                with tf.name_scope("cost_function"):
                    if("SOFT" in self.args.rnn_type):
                        target = self.soft_labels
                        ce = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                logits=self.output_pos,
                                                labels=tf.stop_gradient(target))
                        self.cost = tf.reduce_mean(ce)
                    else:
                        self.hinge_loss = tf.maximum(0.0,(
                                self.args.margin - self.output_pos \
                                + self.output_neg))

                        self.cost = tf.reduce_sum(self.hinge_loss)

                    with tf.name_scope('regularization'):
                        if(self.args.l2_reg>0):
                            vars = tf.trainable_variables()
                            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars \
                                                if 'bias' not in v.name ])
                            lossL2 *= self.args.l2_reg
                            self.cost += lossL2
                    with tf.name_scope('center_loss'):
                        if(self.args.cl_lambda>0):
                            cl_loss = self.cl_loss * self.args.cl_lambda
                            self.cost += cl_loss

                    tf.summary.scalar("cost_function", self.cost)


                if(self.args.dev_lr>0):
                    lr = self.learn_rate
                else:
                    if(self.args.decay_steps>0):
                        lr = tf.train.exponential_decay(self.args.learn_rate,
                                      self.global_step,
                                      self.args.decay_steps,
                                       self.args.decay_lr,
                                       staircase=self.args.decay_stairs)
                    elif(self.args.decay_lr>0 and self.args.decay_epoch>0):
                        decay_epoch = self.args.decay_epoch
                        lr = tf.train.exponential_decay(self.args.learn_rate,
                                      self.global_step,
                                      decay_epoch * self.args.batch_size,
                                       self.args.decay_lr, staircase=True)

                    else:
                        lr = self.args.learn_rate

                if(self.args.cl_lambda>0):
                    control_deps = [self.cl_op]
                else:
                    control_deps = []

                with tf.name_scope('optimizer'):
                    if(self.args.opt=='SGD'):
                        self.opt = tf.train.GradientDescentOptimizer(
                            learning_rate=lr)
                    elif(self.args.opt=='Adam'):
                        self.opt = tf.train.AdamOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='Adadelta'):
                        self.opt = tf.train.AdadeltaOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='Adagrad'):
                        self.opt = tf.train.AdagradOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='RMS'):
                        self.opt = tf.train.RMSPropOptimizer(
                                    learning_rate=lr)
                    elif(self.args.opt=='Moment'):
                        self.opt = tf.train.MomentumOptimizer(lr, 0.9)
                    elif(self.args.opt=='Adamax'):
                        self.opt = AdamaxOptimizer(lr)
                    elif(self.args.opt=='AmsGrad'):
                        print("*Using AMSgrad*")
                        self.opt = AMSGrad(learning_rate=lr)

                    # Use SGD at the end for better local minima
                    self.opt2 = tf.train.GradientDescentOptimizer(
                            learning_rate=self.args.wiggle_lr)
                    tvars = tf.trainable_variables()
                    def _none_to_zero(grads, var_list):
                        return [grad if grad is not None else tf.zeros_like(var)
                              for var, grad in zip(var_list, grads)]
                    if(self.args.clip_norm>0):
                        grads, _ = tf.clip_by_global_norm(
                                        tf.gradients(self.cost, tvars),
                                        self.args.clip_norm)
                        with tf.name_scope('gradients'):
                            gradients = self.opt.compute_gradients(self.cost)
                            if('HYPER' in self.args.rnn_type):
                                # convert riemanian gradients to euclidean gradiens
                                print("Scaling gradients..")
                                gradients = [(H2E_ball(grad),var) for grad,var in gradients]

                            # Log gradients :p
                            # grads_hist = [tf.summary.histogram("grads_{}".format(i), k) for i, k in enumerate(gradients) if k is not None]
                            def ClipIfNotNone(grad):
                                if grad is None:
                                    return grad
                                grad = tf.clip_by_value(grad, -10, 10, name=None)
                                return tf.clip_by_norm(grad, self.args.clip_norm)
                            if(self.args.clip_norm>0):
                                clip_g = [(ClipIfNotNone(grad), var) for grad, var in gradients]
                            else:
                                clip_g = [(grad,var) for grad,var in gradients]


                        # Control dependency for center loss
                        with tf.control_dependencies(control_deps):
                            self.train_op = self.opt.apply_gradients(clip_g,
                                                global_step=self.global_step)
                            self.wiggle_op = self.opt2.apply_gradients(clip_g,
                                                global_step=self.global_step)
                    else:
                        if('HYPER' in self.args.rnn_type):
                            with tf.name_scope('gradients'):
                                gradients = self.opt.compute_gradients(self.cost)
                                if('HYPER' in self.args.rnn_type):
                                    # convert riem opt_fns[opt](params, grads, lr, partial(lr_schedules[lr_schedule], warmup=lr_warmup), n_updates_total, l2=l2, max_grad_norm=max_grad_norm, vector_l2=vector_l2, b1=b1, b2=b2, e=e)anian gradients to euclidean gradiens
                                    print("Scaling gradients..")
                                    gradients = [(H2E_ball(grad),var) for grad,var in gradients]
                            # Control dependency for center loss
                            with tf.control_dependencies(control_deps):
                                self.train_op = self.opt.apply_gradients(gradients,
                                                    global_step=self.global_step)
                                self.wiggle_op = self.opt2.apply_gradients(gradients,
                                                    global_step=self.global_step)
                        else:

                            with tf.control_dependencies(control_deps):
                                self.train_op = self.opt.minimize(self.cost)
                                self.wiggle_op = self.opt2.minimize(self.cost)

                self.grads = _none_to_zero(tf.gradients(self.cost,tvars), tvars)
                # grads_hist = [tf.summary.histogram("grads_{}".format(i), k) for i, k in enumerate(self.grads) if k is not None]
                self.merged_summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
                # model_stats()

                print(self.output_pos)

                 # for Inference
                self.predict_op = self.output_pos
                if('RAW_MSE' in self.args.rnn_type):
                    self.predict_op = tf.clip_by_value(self.predict_op, 1, 5)
                if('SOFT' in self.args.rnn_type):
                    if('POINT' in self.args.rnn_type):
                        predict_neg = 1 - self.predict_op
                        self.predict_op = tf.concat([predict_neg,
                                         self.predict_op], 1)
                    else:
                        self.predict_op = tf.nn.softmax(self.output_pos)
                    self.predictions = tf.argmax(self.predict_op, 1)
                    self.correct_prediction = tf.equal(tf.argmax(self.predict_op, 1),
                                                    tf.argmax(self.soft_labels, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                                                    "float"))
                elif('EXP_MSE' in self.args.rnn_type):
                    self.accuracy = self.cost
                    self.predict_op = tf.exp(-self.output_pos)
                    self.predictions = self.output_pos
                elif('SIG_MSE' in self.args.rnn_type):
                    self.predict_op = tf.nn.sigmoid(self.output_pos)
                    self.predictions = self.predict_op
