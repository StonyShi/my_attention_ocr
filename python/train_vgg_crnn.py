import collections
import logging,re,os
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags
from tensorflow.contrib.tfprof import model_analyzer

import data_provider
import common_flags

from utils import read_charset
import inception_preprocessing
from data_provider import preprocess_image

from vgg_crnn import Vgg

FLAGS = flags.FLAGS
common_flags.define()


flags.DEFINE_string('dict_text',
                   'resource/new_dic2.txt',
                   'absolute path of chinese dict txt')


# yapf: disable
flags.DEFINE_integer('task', 0,
                     'The Task ID. This value is used when training with '
                     'multiple workers to identify each worker.')

flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then'
                     ' the parameters are handled locally by the worker.')

flags.DEFINE_integer('save_summaries_secs', 60,
                     'The frequency with which summaries are saved, in '
                     'seconds.')

flags.DEFINE_integer('save_interval_secs', 600,
                     'Frequency in seconds of saving the model.')

flags.DEFINE_integer('max_number_of_steps', int(1e5),
                     'The maximum number of gradient steps.')

#'./resource/inception_v3.ckpt'
flags.DEFINE_string('checkpoint_inception', '',
                    'Checkpoint to recover inception weights from.')

flags.DEFINE_float('clip_gradient_norm', 2.0,
                   'If greater than 0 then the gradients would be clipped by '
                   'it.')

flags.DEFINE_bool('sync_replicas', False,
                  'If True will synchronize replicas during training.')

flags.DEFINE_integer('replicas_to_aggregate', 1,
                     'The number of gradients updates before updating params.')

flags.DEFINE_integer('total_num_replicas', 1,
                     'Total number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_boolean('reset_train_dir', False,
                     'If true will delete all files in the train_log_dir')

flags.DEFINE_boolean('show_graph_stats', False,
                     'Output model size stats to stderr.')

flags.DEFINE_float('pre_gpu_mem', 0.65,
                   'per_process_gpu_memory_fraction')

# yapf: enable
tf.app.flags.DEFINE_string('height_and_width', '32, 100', 'input size of each image in model training')


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

def prepare_training_dir():
  if not tf.gfile.Exists(FLAGS.train_log_dir):
    logging.info('Create a new training directory %s', FLAGS.train_log_dir)
    tf.gfile.MakeDirs(FLAGS.train_log_dir)
  else:
    if FLAGS.reset_train_dir:
      logging.info('Reset the training directory %s', FLAGS.train_log_dir)
      tf.gfile.DeleteRecursively(FLAGS.train_log_dir)
      tf.gfile.MakeDirs(FLAGS.train_log_dir)
    else:
      logging.info('Use already existing training directory %s',
                   FLAGS.train_log_dir)


def reverse_dict(m_dict):
    return dict(zip(m_dict.values(), m_dict.keys()))

def read_dict(filename, null_character=u'\u2591'):
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                charset[" "] = 0
                logging.warning('incorrect charset file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)  # .decode('utf-8')
            if char == '<nul>':
                char = null_character
            # charset[code] = char
            charset[char] = code
    return charset

def main(_):
    prepare_training_dir()

    chinese_dict = read_dict(FLAGS.dict_text)
    chinese_dict_ids = reverse_dict(chinese_dict)

    device_setter = tf.train.replica_device_setter(
        FLAGS.ps_tasks, merge_devices=True)
    with tf.device(device_setter):
        dataset_name_files = "%s*" % os.path.join(FLAGS.dataset_dir, FLAGS.dataset_name)

        # outs = get_dataset()
        print("-------------------------")
        print(">>> outs: ")
        print(dataset_name_files)
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
        split_results = FLAGS.height_and_width.split(',')
        define_height = int(split_results[0].strip())
        define_width = int(split_results[1].strip())

        width, height, channels = features["width"], features["height"], features["channels"]
        img = tf.decode_raw(features["image"], tf.uint8)

        char_ids = tf.cast(features['char_ids'], tf.int32)
        image_orig = tf.reshape(img, (define_height, define_width, 3))
        # img.set_shape([height, width, channels])

        text = tf.cast(features['text'], tf.string)

        # img, text, char_ids = read_tfrecord("datasets/training.tfrecords", 1, True)
        # img = inception_preprocessing.distort_color(img, random.randrange(0, 4), fast_mode=False, clip=False)

        image = preprocess_image(image_orig, augment=True, num_towers=4)
        image = tf.image.rgb_to_grayscale(image)

        img_batch, text_batch, ids_batch = tf.train.shuffle_batch([image, text, char_ids],
                                                                  batch_size=8,
                                                                  num_threads=8,
                                                                  capacity=3000,
                                                                  min_after_dequeue=1000)

    num_classes = len(chinese_dict.keys())

    vgg = Vgg().build(img_batch, ids_batch, num_classes)

    vgg.print_network()

    loss = vgg.get_loss()

    charset = read_charset(FLAGS.dict_text)
    vgg.summary(ids_batch, charset)

    init_fn = vgg.create_init_fn_to_restore(FLAGS.checkpoint)

    # FLAGS.momentum #FLAGS.optimizer #FLAGS.learning_rate
    optimizer = create_optimizer(FLAGS.learning_rate, FLAGS.optimizer, FLAGS.momentum)
    if FLAGS.sync_replicas:
        replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
        optimizer = tf.train.SyncReplicasOptimizer(opt=optimizer,
                                                   replicas_to_aggregate=FLAGS.replicas_to_aggregate,
                                                   total_num_replicas=FLAGS.total_num_replicas)
        sync_optimizer = optimizer
        startup_delay_steps = 0
    else:
        startup_delay_steps = 0
        sync_optimizer = None

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.pre_gpu_mem
    train_op = slim.learning.create_train_op(
        loss,
        optimizer,
        summarize_gradients=True,
        clip_gradient_norm=FLAGS.clip_gradient_norm)

    slim.learning.train(
        train_op=train_op,
        logdir=FLAGS.train_log_dir,
        graph=loss.graph,
        master=FLAGS.master,
        is_chief=(FLAGS.task == 0),
        number_of_steps=FLAGS.max_number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        startup_delay_steps=startup_delay_steps,
        sync_optimizer=sync_optimizer,
        init_fn=init_fn,
        session_config=tf_config)

# python train_vgg_crnn.py  --dataset_dir=datasets/vgg_train  --dataset_name=train --train_log_dir=vgg_logs > vgg_output.log 2>&1 &
if __name__ == '__main__':
    print("-------------------------------")
    print(FLAGS)
    print("-------------------------------")
    app.run()
