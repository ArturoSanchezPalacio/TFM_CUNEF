import os
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.training.distribute import DistributionStrategy
from tensorflow.contrib.distribute import (MirroredStrategy,
                                           ParameterServerStrategy,
                                           CollectiveAllReduceStrategy)
from time import time

from estimator import CustomEstimator
from dataset import dataset_fn, dataset_size
from model import model_fn
from metrics import metrics_fn, log_fn
from loss import loss_fn
from optimizer import optimizer_fn
import time_utils
import polyaxon_utils


def get_available_gpus():
    """
    :return: The number of available GPUs in the current machine .
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_available_tpus():
    """
    :return: The number of available TPUs in the current machine .
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'TPU']


def get_training_strategy(args):
    if args.train_strategy == 'Distribution':
        return DistributionStrategy()
    elif args.train_strategy == 'Mirrored':
        return MirroredStrategy()
    elif args.train_strategy == 'ParameterServer':
        return ParameterServerStrategy()
    elif args.train_strategy == 'CollectiveAllReduce':
        return CollectiveAllReduceStrategy()
    else:
        return None


def calculate_steps_per_epoch(args, config):
    size = dataset_size(args)
    count_one_tower = int(float(size) / args.batch_size + 0.5)
    gpus_per_node = len(get_available_gpus())
    if gpus_per_node > 1 and config.train_distribute.__class__.__name__ is 'DistributionStrategy':
        gpus_per_node = 1
    if gpus_per_node == 0:
        # if we don't have GPU we count 1 for the CPUs
        gpus_per_node = 1
    return count_one_tower / (gpus_per_node * config.num_worker_replicas)


def train(args):

    # We fix the output directory. Creates a directory in case none is given.
    outputs_dir = args.outputs_dir
    if not tf.gfile.Exists(outputs_dir):
        tf.gfile.MakeDirs(outputs_dir)

    config = tf.estimator.RunConfig(
            model_dir=args.outputs_dir,
            tf_random_seed=args.random_seed,
            train_distribute=get_training_strategy(args),
            log_step_count_steps=args.log_steps,
            save_summary_steps=args.log_steps,
            )

    hooks = []
    # add time hook to stop the training after some time
    if args.max_time is not None:
        hooks.append(StopAtTimeHook(args.max_time))
    # add hook to show a log with different tensors
    hooks.append(PolyaxonMetrics(log_fn(), every_n_iter=args.log_steps))
    # We execute the estimator using all the parameters that have been previously defined:
    estimator = CustomEstimator(
            model_dir=args.outputs_dir,
            model_fn=model_fn,
            optimizer_fn=optimizer_fn,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            params=args,
            config=config,
            steps_per_epoch=calculate_steps_per_epoch(args, config)
            )
    estimator.train(input_fn=lambda: dataset_fn(args), hooks=hooks)


# This method sets a limit on the time of execution:


class StopAtTimeHook(tf.train.SessionRunHook):
    """Hook that requests stop after a specified time."""

    def __init__(self, time_running):
        """
        :param int time_running: Maximum time running
        """
        time_running_secs = time_utils.tdelta(time_running).total_seconds()
        self._end_time = time() + time_running_secs

    def after_run(self, run_context, run_values):
        if time() > self._end_time:
            run_context.request_stop()

class PolyaxonMetrics(tf.train.LoggingTensorHook):
    """Hook that logs data to console and Polyaxon"""

    def __init__(self, tensors_dict, every_n_iter=None, every_n_secs=None):
        super(PolyaxonMetrics, self).__init__(tensors_dict,
                                              every_n_iter=every_n_iter,
                                              every_n_secs=every_n_secs)
        self.tensors_dict = tensors_dict.copy()

    def _log_tensors(self, tensor_values):
        super(PolyaxonMetrics, self)._log_tensors(tensor_values)
        if polyaxon_utils.is_in_cluster():
            for k in self.tensors_dict.keys():
                self.tensors_dict[k] = tensor_values[k]
            polyaxon_utils.send_metrics(**self.tensors_dict)
