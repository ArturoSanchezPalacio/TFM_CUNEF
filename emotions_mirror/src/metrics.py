import tensorflow as tf
import dataset

def metric_variable(shape, dtype, validate_shape=True, name=None):
    return tf.Variable(
            tf.zeros(shape, dtype),
            collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES],
            validate_shape=validate_shape,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.SUM,
            name=name)

def metrics_fn(inputs_dict, model_outputs_dict, outputs_dict, loss, loss_dict, params):
    """
    This function is used to parse the metrics to the estimators during the evaluation

    :param inputs_dict: inputs from the dataset
    :param model_outputs_dict: outputs of the model (predicted values)
    :param outputs_dict: outputs from the dataset (real values)
    :param loss: total loss
    :param loss_dict: a dict with all the losses
    :param params: configuration
    :return: the metrics used for the evaluation
    """

    labels = outputs_dict['outputs']
    predictions = model_outputs_dict['prediction']

    prediction_class = tf.arg_max(predictions, dimension=-1)
    labels = tf.cast(tf.squeeze(labels), dtype=tf.int64)

    if params.mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope('accuracy_metric'):
            accuracy = tf.metrics.accuracy(
                    labels,
                    prediction_class
                    )

        with tf.name_scope('confusion_matrix'):
            num_classes = dataset.get_num_classes(params)
            cm = tf.confusion_matrix(labels, prediction_class, num_classes=num_classes)
            confusion = metric_variable([num_classes, num_classes], tf.int32, name='cm/accumulator')
            confusion_update = confusion.assign_add(cm)
        return {
            'accuracy'        : accuracy,
            'confusion_matrix': (confusion, confusion_update),
            }

    if params.mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope('accuracy_metric'):
            accuracy = tf.contrib.metrics.accuracy(
                    prediction_class,
                    labels,
                    weights=None,
                    name=None
                    )
        accuracy = tf.identity(accuracy, name='metrics/accuracy')
        # Add summary for TensorBoard
        tf.summary.scalar('accuracy', accuracy)

        return {
            'accuracy': accuracy,
            }
    return {}


def log_fn():
    """
    This function is called in the LoggingTensorHook and it returns a dict with the tensors that
    are going to be logged.
    """
    return {
        'loss cross entropy': 'losses/xentropy',
        'epoch'             : 'epoch',
        'learning rate'     : 'optimizer/learning_rate',
        'accuracy'          : 'metrics/accuracy',
        }
