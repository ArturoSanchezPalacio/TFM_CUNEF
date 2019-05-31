#!/usr/bin/env python

import os
import sys
import signal
import argparse
import random
import types
import logging

# Enables a ctr-C without triggering errors
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

# Set logger formatter for tensorflow

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s.%(msecs)03d -- %(levelname)s:%(name)s  %(message)s",
                    datefmt='%y-%m-%d %H:%M:%S')

# Suppress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import polyaxon_utils
import train
import test
import export_model

# The following class is only for reading. It does not have attributes and once you haved edited it you cannot modify
# it, only read it. Assigns a variable creating it if it is not given. If it's given it only allows reading it not
# rewriting it. It's somehow like dynamic definition of constants:


class Configuration(argparse.Namespace):
    """
    Class to store the configuration read with argparse. This file only allow to set once a
    value per key. So the attributes cannot be modified one they are set. If you want to do it you
    have to copy the Configuration and set the new values in the copy method arguments.
    """

    def __init__(self, **kwargs):
        super(Configuration, self).__init__(**kwargs)

    def set(self, name, value):
        if name in self.__dict__ and value != self.__dict__[name]:
            raise Exception('Params cannot be modified')
        else:
            self.__dict__[name] = value

    def __eq__(self, other):
        if not isinstance(other, Configuration):
            return NotImplemented
        return vars(self) == vars(other)

    def __ne__(self, other):
        if not isinstance(other, Configuration):
            return NotImplemented
        return not (self == other)

    def __contains__(self, key):
        return key in self.__dict__

    def __setattr__(self, name, value):
        self.set(name, value)

    def __setitem__(self, name, value):
        self.set(name, value)

    def __delattr__(self, name):
        raise Exception('Params cannot be deleted')

    def __delitem__(self, name):
        raise Exception('Params cannot be deleted')

    def copy(self, **kwargs):
        new_params = self.__dict__.copy()
        for key, value in kwargs.iteritems():
            new_params[key] = value
        return Configuration(**new_params)


if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('command', metavar='COMMAND', choices=['train', 'test', 'export'])
    parser.add_argument('--log_steps', type=int, default=1000, help='Steps to print/save the logs')



    parser.add_argument('--database', type=str, choices=['FER', 'RafD'], default='RafD',
                        help='Database used for training the model.')

    parser.add_argument('--model', type=str, choices=['Alex'], default='Alex',
                        help='Architecture for the network')

    # directories and files
    parser.add_argument('--data', type=str, default="./data/fer2013/fer2013.csv",
                        help='Location of data. When it is specified the dataset will use this file'
                             'to load the data, otherwise it will use the generator dataset. '
                             'By default is None.')

    parser.add_argument('--outputs_dir', type=str,
                        default=polyaxon_utils.get_output_path('./outputs'),
                        help='Location where checkpoints and summaries are saved. '
                             'By default is ./outputs')

    parser.add_argument('--export_dir', type=str,
                        default=polyaxon_utils.get_output_path('./outputs'),
                        help='Location to store the exported model')

    # training hyperparameters

    parser.add_argument('--train_strategy', default=None,
                        choices=['Distribution', 'ParameterServer', 'Mirrored',
                                 'CollectiveAllReduce', None])
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Minibatch size')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='adam',
                        help='Optimizer for training')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.92,
                        help='Learning rate decay per epoch')

    # train parameters
    parser.add_argument('--train_shuffle_size', type=int, default=10000,
                        help='Buffer size of the shuffle data during training')
    parser.add_argument('--max_time', type=str, default=None,
                        help='Maximum allowed time for training. For example: 2hours 30min. '
                             'By default is not set.')

    # reproducible experiments
    parser.add_argument('--random_seed', type=int, default=random.randrange(sys.maxsize),
                        help='Random seed for reproducible experiments')

    args = parser.parse_args(namespace=Configuration)

    args_vars = vars(args)
    args_vars_list = []
    for k in sorted(args_vars.keys()):
        if not (isinstance(args_vars[k], types.FunctionType) or k.startswith('__')):
            args_vars_list.append('{}: {}'.format(k, args_vars[k]))
    tf.logging.info('Using command args: { %s }', ', '.join(args_vars_list))
    args.image_size = 224
    if args.command == 'train':
        args.mode = tf.estimator.ModeKeys.TRAIN
        train.train(args)
    elif args.command == 'test':
        args.mode = tf.estimator.ModeKeys.EVAL
        test.test(args)
    elif args.command == 'export':
        args.mode = tf.estimator.ModeKeys.PREDICT
        export_model.export(args)
    else:
        raise Exception('Wrong command')
