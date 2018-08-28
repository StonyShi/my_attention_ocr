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

    dataset_name_files = "%s*" % os.path.join("datasets/vgg_train", "train")

    # outs = get_dataset()
    print("-------------------------")
    print(">>> outs: ")
    print(dataset_name_files)
    # print(outs)
    print("-------------------------")
    files = tf.train.match_filenames_once(dataset_name_files)
    filename_queue = tf.train.string_input_producer(files)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "image": tf.FixedLenFeature([], tf.string),
        'text': tf.FixedLenFeature([], tf.string),
        'width': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
        'char_ids': tf.VarLenFeature(tf.int64)
    })

    # 设定的resize后的image的大小
    split_results = '32,100'.split(',')
    define_height = int(split_results[0].strip())
    define_width = int(split_results[1].strip())

    width, height, channels = features["width"], features["height"], features["channels"]
    img = tf.decode_raw(features["image"], tf.uint8)

    char_ids = tf.cast(features['char_ids'], tf.int32)
    img = tf.reshape(img, (define_height, define_width, 3))
    # img.set_shape([height, width, channels])

    text = tf.cast(features['text'], tf.string)

    myfont = fm.FontProperties(fname="fonts/card-id.TTF")
    # img, text, char_ids = read_tfrecord("datasets/training.tfrecords", 1, True)
    img = preprocess_image(img, augment=True, num_towers=4)
    # img = inception_preprocessing.distort_color(img, random.randrange(0, 4), fast_mode=False, clip=False)
    img = tf.image.rgb_to_grayscale(img)
    img_batch, text_batch, ids_batch = tf.train.shuffle_batch([img, text, char_ids],
                                                              batch_size=8,
                                                              num_threads=8,
                                                              capacity=3000,
                                                              min_after_dequeue=1000)

    with tf.Session() as sess:

        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        sess.run(init)
        print("--------------------------")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            imgs, texts, ids = sess.run([img_batch, text_batch, ids_batch])

            print(imgs.shape)
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)

