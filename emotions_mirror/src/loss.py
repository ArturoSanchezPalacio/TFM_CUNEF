import tensorflow as tf
from dataset import get_num_classes


def loss_fn(inputs_dict, model_outputs_dict, outputs_dict, params):
    """
    :param inputs_dict: inputs from the dataset
    :param model_outputs_dict: outputs of the model (predicted values)
    :param outputs_dict: outputs from the dataset (real values)
    :param params: configuration
    :return: a tuple with the total loss and a dict with all the losses
    """
    logits = model_outputs_dict['logits']
    real_values = outputs_dict['outputs']
    tf.summary.image('image', inputs_dict['inputs'])

    with tf.name_scope('losses'):
        real_values = tf.squeeze(real_values)
        one_hot_labels = tf.one_hot(real_values, depth=get_num_classes(params))
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels,logits=logits)
        loss = tf.reduce_mean(loss)
        loss = tf.identity(loss, name='xentropy')

    # Add summary for TensorBoard
    tf.summary.scalar('loss/xentropy', loss)

    # in this case the total loss is the same as loss_mse but we could have more than one loss
    loss_total = loss

    return loss_total, {
        'loss_xentropy': loss,
        }
