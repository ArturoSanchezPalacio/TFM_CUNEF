import polyaxon_client
from polyaxon_client.tracking import (get_outputs_path, get_data_paths, Experiment)

import tensorflow as tf

def is_in_cluster():
    return polyaxon_client.settings.IN_CLUSTER

def get_output_path(alternative):
    if not is_in_cluster():
        return alternative
    output_path = get_outputs_path()
    if output_path is None:
        output_path = alternative
    if not tf.gfile.Exists(output_path):
        tf.gfile.MakeDirs(output_path)
    return output_path


def get_data_path(alternative):
    if not is_in_cluster():
        return alternative
    data_path = alternative
    data_paths = get_data_paths()
    if data_paths is not None and 'data' in data_paths:
        data_path = data_paths['data']
    return data_path


def send_metrics(**metrics):
    experimet = Experiment()
    experimet.log_metrics(**metrics)