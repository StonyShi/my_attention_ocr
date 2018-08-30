from Config import Config, GenImage, Progbar, sparse_tuple_from_label,ctc_sparse_from_label,\
    print_net,print_net_line,calculate_distance,get_gb2312,get_gb2312_file,get_logger,decode_sparse_tensor,\
    add_erode,add_dilate,add_noise,add_rotate2

from ImageFile import ImageFileIterator,sparse_tuple_from_label2

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import os,logging
import cv2
from math import *
import string, os, json, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from text.string_generator import (
    create_strings_from_file,
    create_strings_from_new,
    create_strings_from_wikipedia
)

from text.gen_letter import read_dict, reverse_dict

import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.python.platform import test


import inception_preprocessing
from data_provider import preprocess_image
from utils import read_charset, CharsetMapper, decode_code, encode_code

if __name__ == '__main__':

    collection_name = 'my_end_points'
    x = tf.placeholder(dtype=tf.float32, shape=(None, 32, None, 1), name="X")
    is_train = True
    def add_net_collection(net):
        tf.add_to_collection(collection_name, net)


    norm_params = {
        'is_training': is_train,
        'decay': 0.997,
        'epsilon': 1e-05,
        'scale': True
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        padding='SAME',
                        outputs_collections=collection_name,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(0.0001)), \
         slim.arg_scope([slim.max_pool2d, slim.dropout, slim.flatten],
                        outputs_collections=collection_name), \
         slim.arg_scope([slim.batch_norm],
                        decay=0.997, epsilon=1e-5, scale=True, is_training=is_train):


        # (N, 16, 50, 64)
        net = slim.conv2d(x, 64, kernel_size=[3, 3], scope="conv1")
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

        #(8, 8, 25, 128)
        net = slim.conv2d(net, 128, kernel_size=[3, 3], scope="conv2")
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')

        #(8, 4, 25, 256)
        net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 1], stride=[2, 1], scope='pool3')
        ## [kernel_height, kernel_width], [stride_height, stride_width]

        #(8, 2, 25, 512)
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=norm_params):
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 1], stride=[2, 1], scope='pool4')

        #(8, 1, 24, 512)
        net = slim.conv2d(net, 512, kernel_size=[2, 2], stride=1, padding='VALID', scope="conv5")



    print_net_line()
    for net in (tf.get_collection(collection_name)):
        print_net(net)
    print_net_line()

    dataset = np.arange(32 * 100 * 8).reshape((8, 32, 100, 1))
    with tf.Session() as sess:
        # 初始化变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        feed_dict = {x: dataset}
        print(sess.run(net, feed_dict=feed_dict).shape)



