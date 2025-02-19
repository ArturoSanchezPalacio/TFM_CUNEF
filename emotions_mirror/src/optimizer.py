import tensorflow as tf


def optimizer_fn(loss, loss_dict, params, steps_per_epoch):
    """
    :param loss: the total loss
    :param loss_dict: a dictionary with all the losses
    :param params: configuration
    :param steps_per_epoch: number of steps per epoch
    :return: the train operator (An Operation that updates the variables in `var_list`. If `global_step`)
    """
    with tf.variable_scope('optimizer'):
        # set the learning rate to exponential decay learning rate
        learning_rate = tf.train.exponential_decay(params.learning_rate,
                                                   global_step=tf.train.get_global_step(),
                                                   decay_steps=steps_per_epoch,
                                                   decay_rate=params.learning_rate_decay,
                                                   staircase=True, name='lr')

        learning_rate = tf.identity(learning_rate, name='learning_rate')
        # log the learning rate in TensorBoard
        tf.summary.scalar('learning_rate', learning_rate)

        # create the optimizer
        if params.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif params.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        # create the training op
        train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

    return train_op
