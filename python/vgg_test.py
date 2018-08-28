from Config import Config, GenImage, Progbar, sparse_tuple_from_label,ctc_sparse_from_label,\
    print_net,print_net_line,calculate_distance,get_gb2312,get_gb2312_file,get_logger,decode_sparse_tensor,\
    add_erode,add_dilate,add_noise,add_rotate2

from ImageFile import ImageFileIterator,sparse_tuple_from_label2
import matplotlib.pyplot as plt
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

from utils import CharsetMapper

if __name__ == '__main__':

    IMAGE_WIDTH = 100
    IMAGE_HEIGHT = 32

    config = Config(gb2312=True)
    gen_img = GenImage(config=config, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, max_size=11, min_size=10, fonts="./fonts")
    print(gen_img.fonts)
    batch_size = 64
    bg_img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (65, 105, 225))

    image, text = gen_img.gen_image(text_color=(1, 1, 1), bg_img=bg_img)

    print("text: %s,len: %d" % (text, len(text)))


    def get_gen_img():
        image, text = gen_img.gen_image()
        if random.random() > 0.50:
            image = add_rotate2(image)
        image = image.resize((IMAGE_WIDTH, 32), Image.ANTIALIAS)
        image = np.array(image.convert('L'))
        #shape = image.shape
        # print(image.shape)
        if random.random() > 0.50:
            image = add_erode(image)
        if random.random() > 0.50:
            image = add_noise(image)
        if random.random() > 0.50:
            image = add_dilate(image)
        return image, text


    def get_batch_img(batch_size=64):
        # w,h
        batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
        ##print("batch_x: ", batch_x.shape)
        labels = []
        labels_src = []
        for i in range(batch_size):
            image, text = get_gen_img()
            ids = config.text_to_ids(text)
            batch_x[i, :] = image.flatten() / 255.0
            labels_src.append(text)
            labels.append(ids)
        return batch_x, labels, labels_src


    def reformat(dataset, labels, labels_src):
        #dataset = dataset.reshape((-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)).astype(np.float32)
        dataset = dataset.reshape((-1, IMAGE_HEIGHT,  IMAGE_WIDTH, 1)).astype(np.float32)
        labels = sparse_tuple_from_label(labels)
        ##print("dataset: ", dataset.shape)
        return dataset, labels, labels_src


    def next_batch(batch_size=64):
        dataset, labels, labels_src = get_batch_img(batch_size=batch_size)
        dataset, labels, labels_src = reformat(dataset, labels, labels_src)
        return dataset, labels, labels_src


    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="X")
    tf.summary.image('input', x, 5)
    is_train = tf.placeholder(tf.bool, name="is_train")
    y = tf.sparse_placeholder(tf.int32, name='Y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
    print_net(x)
    print_net_line()

    collection_name = 'my_end_points'


    def add_net_collection(net):
        tf.add_to_collection(collection_name, net)


    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        padding='SAME',
                        outputs_collections=collection_name,
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(0.0001)), \
         slim.arg_scope([slim.max_pool2d, slim.dropout, slim.flatten],
                        outputs_collections=collection_name), \
         slim.arg_scope([slim.batch_norm],
                        decay=0.997, epsilon=1e-5, scale=True, is_training=is_train):

        net = slim.conv2d(x, 64, kernel_size=[3, 3], scope="conv1")
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')

        net = slim.conv2d(net, 128, kernel_size=[3, 3], scope="conv2")
        net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')

        net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], stride=[2, 1], scope='pool3')
        # [kernel_height, kernel_width],[stride_height, stride_width]

        net = slim.conv2d(net, 512, kernel_size=[3, 3], scope="conv4")
        net = slim.batch_norm(net, is_training=is_train, activation_fn=None, scope="conv4_batch")
        add_net_collection(net)

        net = slim.conv2d(net, 512, kernel_size=[3, 3], scope="conv5")
        net = slim.batch_norm(net, is_training=is_train, activation_fn=None, scope="conv5_batch")
        add_net_collection(net)

        net = slim.max_pool2d(net, [2, 2], stride=[2, 1], scope='pool4')
        # net = slim.dropout(net, keep_prob=keep_prob, is_training=is_train, scope="pool4/dropout")

        # [kernel_height, kernel_width],[stride_height, stride_width]
        net = slim.conv2d(net, 512, kernel_size=[2, 2], stride=1, padding='VALID', scope="conv6")

        # net = slim.batch_norm(net, is_training=is_train, activation_fn=tf.nn.relu, scope="conv6_batch")
        # add_net_collection(net)

    ## [batch, height, width, features] > [batch, width, height, features]
    net = tf.transpose(net, [0, 2, 1, 3])
    shape = tf.shape(net)  # [batch, width, height, features]
    n, w = shape[0], shape[1]
    # name='seq_len'
    seq_len = tf.ones([n], dtype=tf.int32) * w
    # [batch, width, height, features] > [batch, width, detp]
    cnn_output = tf.reshape(net, [shape[0], -1, 512], name="reshape_cnn_rnn")
    add_net_collection(cnn_output)


    # ## [batch, width, height x features]
    # net = tf.reshape(net, [shape[0], shape[1], shape[2] * shape[3]], name="reshape-cnn")
    # add_net_collection(net)

    def get_lstm_cell2(num_unit, is_train, forget_bias=1.0):
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_unit, forget_bias=forget_bias)
        return tf.where(is_train, tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.75),
                        tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0))


    def get_lstm_cell(num_unit, forget_bias=1.0):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_unit, forget_bias=forget_bias)


    list_n_hidden = [256, 256]

    # Forward direction cells
    fw_cell_list = [get_lstm_cell(size) for size in list_n_hidden]
    # Backward direction cells
    bw_cell_list = [get_lstm_cell(size) for size in list_n_hidden]
    # dropout rnn
    fw_cell_list = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) for cell in fw_cell_list]
    bw_cell_list = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) for cell in bw_cell_list]

    # time_major=False: [batch_size, max_time, depth]
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list
                                                                   , bw_cell_list
                                                                   , inputs=cnn_output
                                                                   , dtype=tf.float32
                                                                   , time_major=False
                                                                   , scope="birnn"
                                                                   )

    add_net_collection(outputs)

    num_classes = config.NUM_CLASSES + 1
    # [batch_size, max_time, depth]
    shape = tf.shape(outputs)
    N, T, D = shape[0], shape[1], shape[2]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, sum(list_n_hidden)], name="reshape-birnn")  # sum(list_n_hidden)
    add_net_collection(outputs)

    # full connection num_classes
    logits_2d = tf.layers.dense(outputs, units=num_classes,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                name='logits_2d')
    add_net_collection(logits_2d)

    # Reshape back to the original shape: [batch_size, max_time, num_classes]
    logits = tf.reshape(logits_2d, [-1, T, num_classes], name="logits")
    add_net_collection(logits)

    logits_raw = tf.argmax(logits, axis=2, name="logits_raw")
    add_net_collection(logits_raw)

    # Convert to time-major: `[max_time, batch_size, num_classes]'
    y_pred = tf.transpose(logits, (1, 0, 2), name="y_pred")
    add_net_collection(y_pred)

    # True: [max_time, batch_size, depth]
    loss = tf.nn.ctc_loss(labels=y, inputs=y_pred, sequence_length=seq_len, time_major=True,
                          ignore_longer_outputs_than_inputs=True)
    add_net_collection(loss)

    safe_loss = tf.where(tf.equal(loss, np.inf), tf.ones_like(loss), loss)
    add_net_collection(safe_loss)

    loss_op = tf.reduce_mean(safe_loss, name="loss_op")
    add_net_collection(loss_op)

    regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
    total_loss = loss_op + regularization_loss

    tf.summary.scalar('ctc_loss', loss_op)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('total_loss', total_loss)

    # beam_width=100, merge_repeated=True
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(y_pred, seq_len)
    # print(decoded) #SparseTensor
    add_net_collection(log_prob)

    dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1, name="dense_decoded")
    add_net_collection(dense_decoded)

    # The error rate
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y), name="acc")
    add_net_collection(acc)


    # y_pred_str,y_str
    def to_spare(dense):
        with tf.name_scope("to_spare"):
            where = tf.not_equal(dense, tf.constant(0, dtype=tf.int64))
            indices = tf.where(where)
            values = tf.gather_nd(dense, indices)
            sparse = tf.SparseTensor(indices, values, dense_shape=tf.shape(dense, out_type=tf.int64))
            return sparse


    logits_raw_spare = to_spare(logits_raw)
    accuracy = tf.reduce_mean(tf.edit_distance(tf.cast(logits_raw_spare, tf.int32), y), name="accuracy")
    add_net_collection(accuracy)
    tf.summary.scalar('accuracy', accuracy)

    for variable in slim.get_model_variables():
        tf.summary.histogram(variable.op.name, variable)

    print_net_line()
    for net in (tf.get_collection(collection_name)):
        print_net(net)

    print_net_line()

    # Training step
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = 1e-6 + tf.train.exponential_decay(1e-3, global_step, decay_steps=1000, decay_rate=0.99,
                                                      staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, name="train_op")
        # train_op2 = slim.learning.create_train_op(cost, optimizer, global_step=global_step)

    tf.logging.set_verbosity(tf.logging.INFO)

    inputs = np.arange(32 * 100 * 8).reshape((8, 32, 100, 1))
    with tf.Session() as sess:
        # 初始化变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        feed_dict = {is_train: True, x: inputs, keep_prob: 0.75}
        print(sess.run(seq_len, feed_dict=feed_dict))
        #     print("net: ", sess.run(net, feed_dict=feed_dict))
        print("cnn_output: ", sess.run(cnn_output, feed_dict=feed_dict).shape)
        print("outputs: ", sess.run(outputs, feed_dict=feed_dict).shape)
        print("logits_2d: ", sess.run(logits_2d, feed_dict=feed_dict).shape)
        print("logits: ", sess.run(logits, feed_dict=feed_dict).shape)
        print("logits_raw: ", sess.run(logits_raw, feed_dict=feed_dict).shape)
        print("y_pred: ", sess.run(y_pred, feed_dict=feed_dict).shape)

    with tf.Session() as sess:
        # 初始化变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        dataset, labels, labels_src = next_batch(8)
        feed_dict = {is_train: True, x: dataset, y: labels, keep_prob: 0.75}
        print("loss: ", sess.run(loss, feed_dict=feed_dict))
        print("safe_loss: ", sess.run(safe_loss, feed_dict=feed_dict))
        print("loss_op: ", sess.run(loss_op, feed_dict=feed_dict))
        print("total_loss: ", sess.run(total_loss, feed_dict=feed_dict))
        print("accuracy: ", sess.run(accuracy, feed_dict=feed_dict))
        # print("edit_distance_op: ", sess.run(edit_distance_op, feed_dict={x: dataset, y: labels}))

    # saver = tf.train.Saver()
    #
    # model_dir = './models/vggcrnn'
    # summaries_dir = './models/vggcrnn/summaries'
    # log_dir = './models/vggcrnn/logs'
    # model_path = './models/vggcrnn/model.ckpt'
    #
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    #
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    #
    # logger = get_logger(log_dir)
    #
    # display_steps = 100
    # total_steps = 10000
    # batch_steps = 2000
    #
    # # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    # # tf_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    # tf_config = tf.ConfigProto(allow_soft_placement=True)
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.75
    #
    # with tf.Session(config=tf_config) as sess:
    #     # Merge all the summaries and write them out to /tmp/logs
    #     merged = tf.summary.merge_all()
    #     train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    #     test_writer = tf.summary.FileWriter(summaries_dir + '/test')
    #     batch_size = 64
    #     start_step = 0
    #     ckpt = tf.train.get_checkpoint_state(model_dir)
    #     if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    #         print("restore last check point: ", ckpt.model_checkpoint_path)
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #     else:
    #         print("not found last check point.")
    #         sess.run(tf.global_variables_initializer())
    #     print("train total_steps:%s, batch_steps:%s, batch_size:%s" % (total_steps, batch_steps, batch_size))
    #     gstep = 0
    #     bar = Progbar(total_steps, batch_steps)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     _global_step = 0
    #     _each = 0
    #     for step in range(start_step + total_steps + 1):
    #         bar.log_begin(step)
    #         for each in range(batch_steps + 1):
    #             _each = each
    #             dataset, labels, labels_src = next_batch(batch_size)
    #             _global_step = (step * batch_steps + each)
    #             feed_dict = {is_train: True, x: dataset, y: labels, keep_prob: 0.75, global_step: _global_step}
    #             _ = sess.run([train_op], feed_dict=feed_dict)
    #             if each % display_steps == 0:
    #                 summary, _, loss, y_predd, acc2 = sess.run([merged, train_op, total_loss, y_pred, accuracy],
    #                                                            feed_dict=feed_dict)
    #                 # _, loss, y_predd, acc2 = sess.run([train_op, cost, y_pred, accuracy], feed_dict=feed_dict)
    #                 train_writer.add_summary(summary, (step * batch_steps + each))
    #                 bar.log_loss(each, loss, acc2)
    #                 logger.info(
    #                     "type=%s,index=%d,loss=%.4f,acc=%.4f" % ("train", (step * batch_steps + each), loss, acc2))
    #         bar.log_end()
    #
    #         dataset, labels, labels_src = next_batch(batch_size)
    #         _global_step = (step * batch_steps + _each)
    #         feed_dict = {is_train: False, x: dataset, y: labels, keep_prob: 1.0, global_step: _global_step}
    #         summary, loss, y_predd, acc2 = sess.run([merged, total_loss, y_pred, accuracy], feed_dict=feed_dict)
    #         test_writer.add_summary(summary, _global_step)
    #         # loss, y_predd, acc2 = sess.run([cost, y_pred, accuracy], feed_dict=feed_dict)
    #         y_pred_str = config.decode_pred(y_predd)
    #         total_distance, total_distance_ed, mean_distance, mean_distance_ed = calculate_distance(labels_src,
    #                                                                                                 y_pred_str)
    #
    #         msg = (">>>Test loss:{},acc:{},t_dist:{},m_dist:{}".format(loss, acc2, total_distance, mean_distance))
    #         print(msg)
    #
    #         logger.info("total_distance: %.4f, total_distance_ed: %.4f, mean_distance: %.4f, mean_distance_ed: %.4f" % (
    #             total_distance, total_distance_ed, mean_distance, mean_distance_ed))
    #         logger.info(
    #             "type=%s,index=%d,loss=%.4f,acc=%.4f" % (
    #             "test", (step * batch_steps + batch_steps), loss, (mean_distance * 100)))
    #
    #         saver.save(sess=sess, save_path=model_path, meta_graph_suffix="meta", write_meta_graph=True, global_step=_global_step)
    #
    #     saver.save(sess=sess, save_path=model_path, meta_graph_suffix="meta", write_meta_graph=True, global_step=_global_step)
    #     coord.request_stop()
    #     coord.join(threads)



