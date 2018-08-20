import os,sys
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def print_node_name(path_ckpt):

    tf.reset_default_graph()
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.75
    with tf.Session(config=tf_config) as sess:
        saver = tf.train.import_meta_graph(path_ckpt + '.meta')
        saver.restore(sess, path_ckpt)

        nodes = [node.name for node in tf.get_default_graph().as_graph_def().node]
        for node in nodes:
            print(node)


def export_model(model_path,output_model, output_node_names=["AttentionOcr_v1/predicted_chars"]):
    if os.path.exists(output_model):
        print("del exists output model: ", output_model)
        os.remove(output_model)
    if not os.path.exists(output_model):
        print("save output model: ", output_model)
        saver = tf.train.Saver()
        #saver = tf.train.import_meta_graph(ckpt + '.meta', clear_devices=True)
        with tf.Session(graph=tf.Graph()) as sess:
            ckpt = tf.train.latest_checkpoint(model_path)
            saver.restore(sess, ckpt)
            output_graph_def = convert_variables_to_constants(sess, sess.graph_def,
                                                              output_node_names=output_node_names)
            with tf.gfile.FastGFile(output_model, mode='wb') as f:
                f.write(output_graph_def.SerializeToString())

def read_output_model(output_model, output_node_names=["AttentionOcr_v1/predicted_chars"], import_name="get"):
    output_node = []
    for o in output_node_names:
        output_node.append("%s:0" % o)
    with open(output_model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        output_node = tf.import_graph_def(graph_def, return_elements=output_node, name="{}/".format(import_name))
    return output_node


if __name__ == '__main__':
    print_node_name('logs/model.ckpt-1426')
    pass