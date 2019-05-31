from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dataset import get_num_classes, name_classes


def model_fn(inputs_dict, mode, params):
    """
    Creates a very simple model with several dense layers
    :param inputs_dict: inputs dict for the dataset
    :param mode: either tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL or
    tf.estimator.ModeKeys.PREDICT
    :param params: configuration
    :return: a dict with the outputs of the model
    """

    num_classes = get_num_classes(params)
    images = inputs_dict['inputs']
    tf.summary.image('images', images)
    if params.model == 'Alex':
        with tf.variable_scope('alex_net'):
            net = images
            net = tf.layers.conv2d(net, 96, 3, 1, activation=tf.nn.relu, name='conv1')
            net = tf.layers.max_pooling2d(net, 2, 2, name='pool1')
            net = tf.layers.conv2d(net, 256, 3, 1, activation=tf.nn.relu, name='conv2')
            net = tf.layers.max_pooling2d(net, 2, 2, name='pool2')
            net = tf.layers.conv2d(net, 384, 3, 1, activation=tf.nn.relu, name='conv3')
            net = tf.layers.conv2d(net, 384, 3, 1, activation=tf.nn.relu, name='conv4')
            net = tf.layers.conv2d(net, 256, 3, 1, activation=tf.nn.relu, name='conv5')
            net = tf.layers.max_pooling2d(net, 2, 2, name='pool3')

            net = tf.layers.flatten(net)
            logits = tf.layers.dense(net, num_classes, activation=None, name='fc_output')
    else:
        raise Exception('unknown model: {}'.format(params.model))

    predictions = tf.nn.softmax(logits, name='predictions')

    return {
        'logits'    : logits,
        'prediction': predictions,
        }
