# Pei, Kexin, et al. "DeepXplore: Automated Whitebox Testing of Deep Learning Systems." (2017).#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import gc

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags



sys.path.append("../")
import sys

from load_model.network import *
from load_model.layer import *

sys.path.append("../")


from coverage_criteria.utils import init_coverage_tables, neuron_covered, update_coverage



FLAGS = flags.FLAGS


def dnn5(input_shape=(None, 13), nb_classes=2):
    """
    The implementation of a DNN model
    :param input_shape: the shape of dataset
    :param nb_classes: the number of classes
    :return: a DNN model
    """
    activation = ReLU
    layers = [Linear(64),
              activation(),
              Linear(32),
              activation(),
              Linear(16),
              activation(),
              Linear(8),
              activation(),
              Linear(4),
              activation(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model


def model_load(datasets):
    tf.reset_default_graph()
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # config.gpu_options.allow_growth = True
    # Create TF session and set as Keras backend session
    sess = tf.Session(config=config)
    print("Created TensorFlow session.")

    input_shape = (None, 13)
    nb_classes = 2
    x = tf.placeholder(tf.float32, shape=input_shape)
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    feed_dict = None

    model = dnn5(input_shape, nb_classes)

    preds = model(x)
    print("Defined TensorFlow model graph.")

    saver = tf.train.Saver()

    model_path = '../mod/' + datasets + '/test.model'

    saver.restore(sess, model_path)

    return sess, preds, x, y, model, feed_dict


def neuron_coverage(datasets, model_name, de=False, attack='fgsm'):
    """
    :param datasets
    :param model
    :param samples_path
    :return:
    """
    # Object used to keep track of (and return) key accuracies
    m = np.load('../data/data-x.npy')
    n = np.load('../data/data-y.npy')
    p = int(m.shape[0] * 0.8)
    X_train = m[:p]
    Y_train = n[:p]
    X_test = m[p:]
    Y_test = n[p:]

    samples = X_test

    n_batches = 10
    X_train_boundary = X_train
    store_path = "../coverage-result/dnn5/adult/"

    for i in range(n_batches):
        print(i)

        tf.reset_default_graph()

        sess, preds, x, y, model, feed_dict = model_load(datasets=datasets)
        model_layer_dict = init_coverage_tables(model)
        model_layer_dict = update_coverage(sess, x, samples, model, model_layer_dict, feed_dict, threshold=0)
        sess.close()
        del sess, preds, x, y, model, feed_dict
        gc.collect()

        result = neuron_covered(model_layer_dict)[2]
        print('covered neurons percentage %d neurons %f'
              % (len(model_layer_dict), result))


def main(argv=None):
    neuron_coverage(datasets=FLAGS.datasets,
                    model_name=FLAGS.model,
                   )


if __name__ == '__main__':
    flags.DEFINE_string('datasets', 'adult', 'The target datasets.')
    flags.DEFINE_string('model', 'dnn5', 'The name of model')


    tf.app.run()
