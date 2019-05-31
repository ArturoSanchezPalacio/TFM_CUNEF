import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
import os

def load_images_RafD(args):
    files_90 = tf.gfile.Glob(os.path.join(args.data, 'Rafd090*.jpg'))
    files_135 = tf.gfile.Glob(os.path.join(args.data, 'Rafd135*.jpg'))
    files_45 = tf.gfile.Glob(os.path.join(args.data, 'Rafd045*.jpg'))
    files = files_90 + files_45 + files_135

    # Create a validation set:

    validation = []
    for img_num in ['01', '03', '19', '29', '64']:
        for rot in ['090', '135', '045']:
            data_path = os.path.join(args.data, 'Rafd{}_{}*.jpg'.format(rot, img_num))
            val_files = tf.gfile.Glob(data_path)
            validation = validation + val_files

    # Create a training set
    train = [f for f in files if f not in validation]

    if args.command == 'train':
        return train
    elif args.command == 'test':
        return validation
    else:
        raise Exception('unknown command: {}'.format(args.command))


def dataset_size(args):
    if args.database == 'FER':
        # file dataset, we count the number of lines without the header line
        with tf.gfile.Open(args.data, 'r') as f:
            size = len(f.readlines()) - 1
    elif args.database == 'RafD':
        files = load_images_RafD(args)
        size = len(files)
    else:
        raise Exception('unknown dataset: {}'.format(args.database))
    return size

def get_num_classes(args):
    """
    :param args:
    :return: number of classes depending on the dataset
    """
    if args.database == 'FER':
        num_classes = 7
    elif args.database == 'RafD':
        num_classes = 8
    else:
        raise Exception('unknown dataset: {}'.format(args.database))
    return num_classes

def name_classes(args):
    """
    :param args:
    :return: list of emotions tagged in the dataset
    """
    if args.database == 'FER':
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    elif args.database == 'RafD':
        emotions = ['angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    else:
        raise Exception('unknown dataset: {}'.format(args.database))
    return emotions


def dataset_fn(args):
    with tf.variable_scope('dataset'):
        if args.database == 'FER':
            return FER_dataset(args)
        elif args.database == 'RafD':
            return RafD_dataset(args)
        else:
            raise Exception('unknown dataset: {}'.format(args.database))


def FER_dataset(args):
    dataset = tf.data.TextLineDataset([args.data])

    # skip header of the file
    dataset = dataset.skip(1)

    if args.mode == tf.estimator.ModeKeys.TRAIN:

        # repeat dataset for the number of epochs
        dataset = dataset.repeat(args.epochs)

        # adds shuffle if training
        dataset = dataset.shuffle(args.train_shuffle_size)

    # process line: map + py_func
    # We parse each line assigning each side of the line to x and y respectively.
    def _map_fn(example_serialized):
        def _parse(line):
            if (not isinstance(line, bytes)):
                line = line.encode()
            line_s = line.split(",".encode())
            if len(line_s) != 3:
                raise Exception('The line must have only 3 elements but it is {}'.format(line))
            x = np.float32(line_s[0])
            cadenas = line_s[1].split()
            array_cadenas = np.array(cadenas)
            y = np.array(array_cadenas, dtype=np.float32)
            return x, y

        outputs, inputs = tf.py_func(_parse,
                                     inp=[example_serialized],
                                     Tout=[tf.float32, tf.float32],
                                     stateful=True)
        # reshape data
        inputs = inputs / 255.0
        inputs = tf.reshape(inputs, [48, 48, 1])
        inputs = tf.image.resize(inputs, [224, 224])
        inputs = tf.image.grayscale_to_rgb(inputs)
        inputs = randomize_image(inputs, horizontal_flip=True, white_noise=[224, 224, 3],
                                 color_distort=True)
        outputs = tf.reshape(outputs, [1])

        return {'inputs': inputs}, {'outputs': outputs}

    # We use the parsing we defined just above to parse the dataset and we divide it in batches:
    dataset = dataset.map(_map_fn)

    # batch size
    dataset = dataset.batch(args.batch_size)
    return dataset


def RafD_dataset(args):
    # generates the name of the files and the code associated to the emotion
    files = load_images_RafD(args)
    dic_emotions = dict(zip(['happy', 'angry', 'sad', 'contemptuous', 'disgusted', 'neutral',
                             'fearful', 'surprised'], list(range(0, 8))))

    def _generator():
        for f1 in files:
            parts = os.path.basename(f1)
            title = parts.split("_")
            emotion = title[4]
            yield [f1], [dic_emotions[emotion]]

    dataset = tf.data.Dataset.from_generator(_generator,
                                             output_types=(tf.string, tf.int32),
                                             output_shapes=([1], [1]))

    # repeat dataset for the number of epochs for training
    if args.mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat(args.epochs)

    # map function to set the inputs and outputs to dictionaries as in the FER_dataset
    def _map_fn(inputs, outputs):
        filename = tf.reshape(inputs,[])
        img_raw = tf.read_file(filename)
        img_tensor = decode_image(img_raw)
        img_tensor = tf.image.central_crop(img_tensor, 0.7)
        img_tensor = tf.image.resize_images(img_tensor, [224, 224])
        img_final = tf.image.convert_image_dtype(img_tensor, dtype=tf.float32)
        img_final = randomize_image(img_final, horizontal_flip=True, white_noise=[224, 224, 3],
                                    color_distort=True)
        inputs = img_final
        return {'inputs': inputs}, {'outputs': outputs}

    dataset = dataset.map(_map_fn)

    # batch size
    dataset = dataset.batch(args.batch_size)

    return dataset


def randomize_image(image, crop=False, resize=None, color_distort=False, vertical_flip=False,
                    horizontal_flip=False, white_noise=None, fast_mode=False, scope=None):
    """
    :param image: Image to randomize
    :param crop: 1-D tensor with size the rank of `value` to random_crop the image.
    :param resize: A number to resize the image or a list of [height, width]
    :param color_distort: Boolean flag to distort randomly the color of image
    :param vertical_flip: Boolean flag to random vertical flip the image
    :param horizontal_flip: Boolean flag to random horizontal flip the image
    :param white_noise: Shape of the noise tensor, None to not to add noise
    :param fast_mode: Boolean flag to apply faster operations in distortion (for resize and color
    only)
    :param scope: Optional scope for name_scope.
    :return: 3-D Tensor color-distorted image on range [0, 1]
    :raises: ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_image', [image]):
        distorted_image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # crop image
        if crop:
            distorted_image = tf.image.random_crop(distorted_image, crop)

        # resize
        if resize:
            num_resize_cases = 1 if fast_mode else 4
            distorted_image = tf.expand_dims(distorted_image, 0)
            distorted_image = apply_with_random_selector(
                    distorted_image,
                    lambda x, method: tf.image.resize_images(x, resize, method=method),
                    num_cases=num_resize_cases)
            distorted_image = tf.squeeze(distorted_image, [0])

        # horizontal/vertical flip
        if horizontal_flip:
            distorted_image = tf.image.random_flip_left_right(distorted_image)
        if vertical_flip:
            distorted_image = tf.image.random_flip_up_down(distorted_image)

        # distort color
        if color_distort:
            num_distort_color_cases = 2 if fast_mode else 4
            distorted_image = apply_with_random_selector(
                    distorted_image,
                    lambda x, ordering: distort_color(x, ordering, fast_mode),
                    num_cases=num_distort_color_cases)

        # add white noise
        if white_noise:
            white_noise_dental = 0.02
            white_noise = tf.random_uniform(white_noise,
                                            -white_noise_dental,
                                            white_noise_dental,
                                            dtype=tf.float32)
            distorted_image = tf.add(distorted_image, white_noise)

        # The random_* ops do not necessarily clamp.
        distorted_image = tf.clip_by_value(distorted_image, 0.0, 1.0)

    return distorted_image


def apply_with_random_selector(input_tensor, func, num_cases):
    """
    Computes func(x, sel), with sel sampled from [0...num_cases-1].
    :param input_tensor: input Tensor.
    :param func: Python function to apply.
    :param num_cases: Python int32, number of cases to sample sel from.
    :return: The result of func(x, sel), where func receives the value of the
  selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge(
            [func(control_flow_ops.switch(input_tensor, tf.equal(sel, case))[1], case) for case in
             range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=False, scope=None):
    """
    Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    :param image: 3-D Tensor containing single image in [0, 1].
    :param color_ordering: Python int, a type of distortion (valid values: 0-3).
    :param fast_mode: Avoids slower ops (random_hue and random_contrast)
    :param scope: Optional scope for name_scope.
    :return: 3-D Tensor color-distorted image on range [0, 1]
    :raises: ValueError: if color_ordering not in [0, 3]
    """
    brightness_delta = 10.0 / 255.0
    saturation_delta = 0.1
    hue_delta = 0.05
    contrast_delta = 0.15
    sat_lower = 1. - saturation_delta
    sat_upper = 1. + saturation_delta
    con_lower = 1. - contrast_delta
    con_upper = 1. + contrast_delta
    if fast_mode:
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=brightness_delta)
            image = tf.image.random_saturation(image, lower=sat_lower, upper=sat_upper)
        else:
            image = tf.image.random_saturation(image, lower=sat_lower, upper=sat_upper)
            image = tf.image.random_brightness(image, max_delta=brightness_delta)
    else:
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=brightness_delta)
            image = tf.image.random_saturation(image, lower=sat_lower, upper=sat_upper)
            image = tf.image.random_hue(image, max_delta=hue_delta)
            image = tf.image.random_contrast(image, lower=con_lower, upper=con_upper)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=sat_lower, upper=sat_upper)
            image = tf.image.random_brightness(image, max_delta=brightness_delta)
            image = tf.image.random_contrast(image, lower=con_lower, upper=con_upper)
            image = tf.image.random_hue(image, max_delta=hue_delta)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=con_lower, upper=con_upper)
            image = tf.image.random_hue(image, max_delta=hue_delta)
            image = tf.image.random_brightness(image, max_delta=brightness_delta)
            image = tf.image.random_saturation(image, lower=sat_lower, upper=sat_upper)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=hue_delta)
            image = tf.image.random_saturation(image, lower=sat_lower, upper=sat_upper)
            image = tf.image.random_contrast(image, lower=con_lower, upper=con_upper)
            image = tf.image.random_brightness(image, max_delta=brightness_delta)
        else:
            raise ValueError('color_ordering must be in [0, 3]')
    return image


def decode_image(image_bytes_tensor, channels=3):
    """We need this function as the tf function does not return a shape"""
    with tf.name_scope('decode_image') as scope:
        substr_jpg = tf.substr(image_bytes_tensor, 0, 3)
        substr_png = tf.substr(image_bytes_tensor, 0, 4)

        def _png():
            is_png = tf.equal(substr_png, b'\211PNG', name='is_png')
            assert_decode = tf.Assert(is_png, ['Unable to decode bytes as JPEG or PNG'])
            with tf.control_dependencies([assert_decode]):
                image_png = tf.image.decode_png(image_bytes_tensor, channels=channels)
                image_png = tf.image.convert_image_dtype(image_png, dtype=tf.float32)
                return image_png

        def _jpeg():
            image_jpg = tf.image.decode_jpeg(image_bytes_tensor, channels=channels)
            image_jpg = tf.image.convert_image_dtype(image_jpg, dtype=tf.float32)
            return image_jpg

        is_jpeg = tf.equal(substr_jpg, b'\xff\xd8\xff', name='is_jpeg')
        return tf.cond(is_jpeg, _jpeg, _png, name='cond_jpeg')