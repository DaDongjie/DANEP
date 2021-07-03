import tensorflow as tf
import os
import pickle
from Utils.utils import *


w_init = lambda:tf.random_normal_initializer(stddev=0.02)


class Model(object):

    def __init__(self, config):
        self.config = config
        self.att_shape = config['att_shape']
        self.att_input_dim = config['att_input_dim']
        self.is_init = config['is_init']
        self.pretrain_params_path = config['pretrain_params_path']
        self.labels = None

        self.num_att_layers = len(self.att_shape)

        if self.is_init:
            if os.path.isfile(self.pretrain_params_path):
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                with open(self.pretrain_params_path, 'rb') as handle:
                    self.W_init, self.b_init = pickle.load(handle)


    def forward_att(self, x, drop_prob, reuse=False):

        with tf.variable_scope('att_encoder', reuse=reuse) as scope:

            cur_input = x
            print(cur_input.get_shape())

            # ============encoder===========
            struct = self.att_shape
            for i in range(self.num_att_layers):
                name = 'att_encoder' + str(i)
                if self.is_init:
                    cur_input = tf.layers.dense(cur_input, units=struct[i],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i], kernel_initializer=w_init())
                if i < self.num_att_layers - 1:
                    cur_input = lrelu(cur_input)
                    cur_input = tf.layers.dropout(cur_input, drop_prob)
                print(cur_input.get_shape())

            att_H = cur_input  # 学到的表示

            # ====================decoder=============
            struct.reverse()
            cur_input = att_H
            for i in range(self.num_att_layers - 1):
                name = 'att_decoder' + str(i)
                if self.is_init:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1], kernel_initializer=w_init())
                cur_input = lrelu(cur_input)
                cur_input = tf.layers.dropout(cur_input, drop_prob)
                print(cur_input.get_shape())

            name = 'att_decoder' + str(self.num_att_layers - 1)
            if self.is_init:
                cur_input = tf.layers.dense(cur_input, units=self.att_input_dim,
                                            kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                            bias_initializer=tf.constant_initializer(self.b_init[name]))
            else:
                cur_input = tf.layers.dense(cur_input, units=self.att_input_dim, kernel_initializer=w_init())

            x_recon = cur_input
            print(cur_input.get_shape())

            self.att_shape.reverse()
        return att_H, x_recon