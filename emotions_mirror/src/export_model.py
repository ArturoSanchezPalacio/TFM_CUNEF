import tensorflow as tf
from dataset import decode_image
from estimator import CustomEstimator
from model import model_fn


def serving_input_receiver_fn(args):
    image_uint = tf.placeholder(tf.uint8, name='input_tensor', shape=[None, None, 3])
    image_float = tf.image.convert_image_dtype(image_uint, dtype=tf.float32)
    image_float = tf.image.resize(image_float, [224, 224])
    images_float = tf.expand_dims(image_float, 0)

    return tf.estimator.export.ServingInputReceiver(
            features={'inputs': images_float},
            receiver_tensors={'inputs': image_uint})

def export(args):

    estimator = CustomEstimator(
        model_dir= args.outputs_dir,
        model_fn = model_fn,
        params = args
    )

    estimator.export_savedmodel(args.export_dir, lambda: serving_input_receiver_fn(args),
                                strip_default_attrs=True)