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

import utils

from utils import read_dict, reverse_dict, read_charset, CharsetMapper, decode_code, encode_code

import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.python.platform import test


def create_optimizer(learning_rate, optimizer_type="adam", momentum=0.8):
    """Creates optimized based on the specified flags."""
    if optimizer_type == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    elif optimizer_type == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer_type == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, momentum=momentum)
    return optimizer



class Vgg(object):

    def __init__(self, charset_dict:dict=None):
        self.charset_dict = charset_dict

    def build(self, x:tf.Tensor, y:tf.SparseTensor, num_classes:int, collection_name = 'my_end_points'):
        self.x = x
        self.y = y
        is_train = y is not None
        self.is_train = is_train
        self.collection_name = collection_name
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
                net = slim.conv2d(net, 512, kernel_size=[3, 3], scope="conv4")
                net = slim.batch_norm(net, is_training=is_train, activation_fn=None, scope="conv4_batch")
                self.add_net_collection(net)
                net = slim.conv2d(net, 512, kernel_size=[3, 3], scope="conv5")
                net = slim.batch_norm(net, is_training=is_train, activation_fn=None, scope="conv5_batch")
                self.add_net_collection(net)
                net = slim.max_pool2d(net, [2, 2], stride=[2, 1], scope='pool4')
                net = slim.conv2d(net, 512, kernel_size=[2, 2], stride=1, padding='VALID', scope="conv6")

        self.cnn_net = net
        self.pool_net = self.pool_views_fn(net)

        # [batch, height, width, features] > [batch, width, height, features]
        net = tf.transpose(net, [0, 2, 1, 3])
        shape = tf.shape(net)  # [batch, width, height, features]
        n, w = shape[0], shape[1]
        seq_len = tf.ones([n], dtype=tf.int32) * w

        # [batch, width, height, features] > [batch, width, depth]
        cnn_output = tf.reshape(net, [shape[0], -1, 512], name="reshape_cnn_rnn")
        self.cnn_output = cnn_output
        self.add_net_collection(cnn_output)

        def get_lstm_cell(num_unit, forget_bias=1.0):
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_unit, forget_bias=forget_bias)

        list_n_hidden = [256, 256]

        # Forward direction cells
        fw_cell_list = [get_lstm_cell(size) for size in list_n_hidden]
        # Backward direction cells
        bw_cell_list = [get_lstm_cell(size) for size in list_n_hidden]
        # dropout rnn
        if is_train:
            fw_cell_list = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.75) for cell in fw_cell_list]
            bw_cell_list = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.75) for cell in bw_cell_list]

        # time_major=False: [batch_size, max_time, depth]
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cell_list
                                                                       , bw_cell_list
                                                                       , inputs=cnn_output
                                                                       , dtype=tf.float32
                                                                       , time_major=False
                                                                       , scope="birnn"
                                                                       )
        self.rnn_output = outputs
        self.add_net_collection(outputs)
        # [batch_size, max_time, depth]
        shape = tf.shape(outputs)
        N, T, D = shape[0], shape[1], shape[2]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, sum(list_n_hidden)], name="reshape-birnn")  # sum(list_n_hidden)
        self.add_net_collection(outputs)

        # full connection num_classes
        logits_2d = tf.layers.dense(outputs, units=num_classes,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    name='logits_2d')
        # logits_2d = slim.fully_connected(outputs, num_classes, activation_fn=None, scope='logits_2d')
        self.add_net_collection(logits_2d)

        # Reshape back to the original shape: [batch_size, max_time, num_classes]
        logits = tf.reshape(logits_2d, [-1, T, num_classes], name="logits")
        self.add_net_collection(logits)

        logits_raw = tf.argmax(logits, axis=2, name="logits_raw")
        self.add_net_collection(logits_raw)

        predicted_ids = tf.to_int32(logits_raw, name='predicted_ids')
        self.add_net_collection(predicted_ids)

        # Convert to time-major: `[max_time, batch_size, num_classes]'
        y_pred = tf.transpose(logits, (1, 0, 2), name="y_pred")
        self.add_net_collection(y_pred)


        self.seq_len = seq_len
        self.outputs = outputs
        self.logits_2d = logits_2d
        self.logits = logits
        self.logits_raw = logits_raw
        self.predicted_ids = predicted_ids
        self.y_pred = y_pred

        # True: [max_time, batch_size, depth]
        loss = tf.nn.ctc_loss(labels=y, inputs=y_pred, sequence_length=seq_len, time_major=True,
                              ignore_longer_outputs_than_inputs=True)
        self.add_net_collection(loss)

        safe_loss = tf.where(tf.equal(loss, np.inf), tf.ones_like(loss), loss)
        # self.add_net_collection(safe_loss)

        loss_op = tf.reduce_mean(loss, name="loss_op")
        self.add_net_collection(loss_op)

        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        total_loss = loss_op + regularization_loss

        # beam_width=100, merge_repeated=True
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(y_pred, seq_len)
        # print(decoded) #SparseTensor
        self.add_net_collection(log_prob)

        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1, name="dense_decoded")
        self.add_net_collection(dense_decoded)

        # The error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y), name="acc")
        self.add_net_collection(acc)

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
        self.add_net_collection(accuracy)

        self.loss = loss
        self.safe_loss = safe_loss
        self.loss_op = loss_op
        self.regularization_loss = regularization_loss
        self.total_loss = total_loss
        self.decoded = decoded
        self.dense_decoded = dense_decoded
        self.acc = acc
        self.accuracy = accuracy
        return self

    def get_loss(self):
        slim.losses.add_loss(self.total_loss)
        total_loss = slim.losses.get_total_loss(False)
        return total_loss

    def create_init_fn_to_restore(self, master_checkpoint):
        """Creates an init operations to restore weights from various checkpoints.

        Args:
          master_checkpoint: path to a checkpoint which contains all weights for
            the whole model.
        Returns:
          a function to run initialization ops.
        """
        all_assign_ops = []
        all_feed_dict = {}

        def assign_from_checkpoint(variables, checkpoint):
            logging.info('Request to re-store %d weights from %s',
                         len(variables), checkpoint)
            if not variables:
                logging.error('Can\'t find any variables to restore.')
                sys.exit(1)
            assign_op, feed_dict = slim.assign_from_checkpoint(checkpoint, variables)
            all_assign_ops.append(assign_op)
            all_feed_dict.update(feed_dict)

        logging.info('variables_to_restore:\n%s' % utils.variables_to_restore().keys())
        logging.info('moving_average_variables:\n%s' % [v.op.name for v in tf.moving_average_variables()])
        logging.info('trainable_variables:\n%s' % [v.op.name for v in tf.trainable_variables()])
        if master_checkpoint:
            assign_from_checkpoint(utils.variables_to_restore(), master_checkpoint)

        def init_assign_fn(sess):
            logging.info('Restoring checkpoint(s)')
            sess.run(all_assign_ops, all_feed_dict)

        return init_assign_fn

    def pool_views_fn(self, nets):
        """Combines output of multiple convolutional towers into a single tensor.

        It stacks towers one on top another (in height dim) in a 4x1 grid.
        The order is arbitrary design choice and shouldn't matter much.

        Args:
          nets: list of tensors of shape=[batch_size, height, width, num_features].

        Returns:
          A tensor of shape [batch_size, seq_length, features_size].
        """
        with tf.variable_scope('pool_views_fn/STCK'):
            net = tf.concat(nets, 1)
            shape = tf.shape(net)
            batch_size = shape[0] #net.get_shape().dims[0].value
            feature_size = shape[3] #net.get_shape().dims[3].value
            return tf.reshape(net, [batch_size, -1, feature_size])

    def test_net_out(self, batch_x):
        charset = read_charset("resource/new_dic2.txt")
        charset_mapper = CharsetMapper(charset)

        with tf.Session() as sess:
            init_op = (tf.global_variables_initializer(), tf.tables_initializer())
            sess.run(init_op)
            feed_dict = {self.x: batch_x}
            run_list = [self.seq_len,
                        self.cnn_output,
                        self.pool_net,
                        self.cnn_net,
                        self.rnn_output,
                        self.outputs,
                        self.logits_2d,
                        self.logits,
                        self.logits_raw,
                        self.y_pred, self.predicted_ids]
            _seq_len, _cnn_output, _pool_net, _cnn_net,_rnn_output, _outputs,_logits_2d, _logits,_logits_raw,_y_pred,_predicted_ids \
                = sess.run(run_list, feed_dict=feed_dict)
            print("seq_len: ", _seq_len)
            print("cnn_output: ", _cnn_output.shape)
            print("cnn_net: ", _cnn_net.shape)
            print("pool_net: ", _pool_net.shape)
            print("rnn_output: ", _rnn_output.shape)
            print("outputs: ", _outputs.shape)
            print("logits_2d: ", _logits_2d.shape)
            print("logits: ", _logits.shape)
            print("logits_raw: ", _logits_raw.shape)
            print("_predicted_ids_last: ", _predicted_ids[-1])
            print("_predicted_text: ", sess.run(charset_mapper.get_text(_predicted_ids[:4,:])))
            # print("_predicted_text: ", sess.run(charset_mapper.get_text(_predicted_ids)))
            print("y_pred: ", _y_pred.shape)

    def add_net_collection(self, net):
        tf.add_to_collection(self.collection_name, net)

    def summary(self, ids=None, charset=None):
        def sname(label):
            prefix = 'train' if self.is_train else 'eval'
            return '%s/%s' % (prefix, label)

        tf.summary.image('input', self.x, 5)
        tf.summary.scalar('ctc_loss', self.loss_op)
        tf.summary.scalar('regularization_loss', self.regularization_loss)
        tf.summary.scalar('total_loss', self.total_loss)

        tf.summary.scalar('accuracy', self.accuracy)

        if ids is not None and charset is not None:
            max_outputs = 4
            charset_mapper = CharsetMapper(charset)

            pr_text = charset_mapper.get_text(self.predicted_ids[:max_outputs, :])
            tf.summary.text(sname('text/pr'), pr_text)

            #de_ids = decode_sparse_tensor(ids)
            de_ids = tf.sparse_to_dense(ids._indices, ids._dense_shape, ids._values)
            gt_text = charset_mapper.get_text(de_ids[:max_outputs,:])
            tf.summary.text(sname('text/gt'), gt_text)

        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)
        pass

    def predict(self):
        pass

    def train_op(self):
        # Training step
        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = 1e-6 + tf.train.exponential_decay(1e-3, global_step, decay_steps=1000, decay_rate=0.90,
                                                          staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(self.total_loss, name="train_op")

        self.learning_rate = learning_rate
        self.global_step = global_step
        self.train_op = train_op
        return self

    def test_loss(self, dataset, labels, labels_src):
        print("-------------")
        ids_batch = tf.SparseTensor(labels[0], labels[1], labels[2])
        print(ids_batch)
        de_ids = tf.sparse_to_dense(ids_batch._indices, ids_batch._dense_shape, ids_batch._values)

        charset = read_charset("resource/new_dic2.txt")
        charset_mapper = CharsetMapper(charset)

        print("-------------")
        with tf.Session() as sess:
            # 初始化变量
            init_op = (tf.global_variables_initializer(), tf.tables_initializer())
            sess.run(init_op)
            feed_dict = {self.x: dataset, self.y: labels}
            print("loss: ", sess.run(self.loss, feed_dict=feed_dict))
            #print("safe_loss: ", sess.run(self.safe_loss, feed_dict=feed_dict))
            print("loss_op: ", sess.run(self.loss_op, feed_dict=feed_dict))
            print("total_loss: ", sess.run(self.total_loss, feed_dict=feed_dict))
            print("accuracy: ", sess.run(self.accuracy, feed_dict=feed_dict))
            print("de_ids_last: ", sess.run(de_ids)[-1])
            print("de_ids_str: ", sess.run(charset_mapper.get_text(de_ids[:4,:])))
            print("labels_src: labels_src")

            #print("acc: ", sess.run(self.acc, feed_dict={x: dataset, y: labels}))


    def print_network(self):
        print_net_line()
        print_net(self.x)
        print_net_line()
        for net in (tf.get_collection(self.collection_name)):
            print_net(net)
        print_net_line()


if __name__ == '__main__':


    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 100

    config = Config(gb2312=True)

    gen_img = GenImage(config=config, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, max_size=11, min_size=10, fonts="./fonts")

    num_classes = config.NUM_CLASSES + 1

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="X")
    y = tf.sparse_placeholder(tf.int32, name='Y')

    vgg = Vgg().build(x, y, num_classes)



    bg_img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (65, 105, 225))

    words = create_strings_from_file("text/cns.txt", 200)
    image, text = gen_img.gen_image(text_color=(1, 1, 1), bg_img=bg_img, words=words)

    print("text: %s,len: %d" % (text, len(text)))

    #image.show()


    def get_gen_img():
        image, text = gen_img.gen_image(words=words)
        if random.random() > 0.50:
            image = add_rotate2(image)
        image = image.resize((IMAGE_WIDTH, 32), Image.ANTIALIAS)
        image = np.array(image.convert('L'))
        #shape = image.shape
        if random.random() > 0.50:
            image = add_erode(image)
        if random.random() > 0.50:
            image = add_noise(image)
        if random.random() > 0.50:
            image = add_dilate(image)
        return image, text


    image, text = get_gen_img()

    # plt.figure()
    # plt.imshow(image)
    # plt.show()

    print("text: %s,len: %d" % (text, len(text)))
    print(image.shape)


    def get_batch_img(batch_size=64):
        # w,h
        batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
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
        dataset = dataset.reshape((-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)).astype(np.float32)
        labels = sparse_tuple_from_label(labels)
        ##print("dataset: ", dataset.shape)
        return dataset, labels, labels_src

    def next_batch(batch_size=64):
        dataset, labels, labels_src = get_batch_img(batch_size)
        dataset, labels, labels_src = reformat(dataset, labels, labels_src)
        return dataset, labels, labels_src


    batch_x = np.arange(32 * 100 * 8).reshape((8, 32, 100, 1))
    vgg.test_net_out(batch_x)

    # vgg.print_network()

    dataset, labels, labels_src = next_batch(8)
    vgg.test_loss(dataset, labels, labels_src)



