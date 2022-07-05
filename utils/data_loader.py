import tensorflow as tf

from PIL import Image

from iseg.data_process.augments.pad_to_odd_augment import pad_to_odd


def load_label_to_tensor(label_path):
    label_tensor = tf.py_function(__load_label_to_tensor_internel, [label_path], tf.int32)
    label_tensor.set_shape([None, None, 1])

    return label_tensor


def __load_label_to_tensor_internel(path_tensor):

    label_path = path_tensor.numpy()
    label_image = Image.open(label_path)
    label_array = tf.keras.preprocessing.image.img_to_array(label_image, "channels_last")

    label_tensor = tf.convert_to_tensor(label_array)
    label_tensor = tf.cast(label_tensor, tf.int32)

    return label_tensor


def load_image_tensor_from_path(image_path, label_path=None):

    image_tensor = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    image_tensor = tf.cast(image_tensor, tf.float32)

    label_tensor = None

    if label_path is not None:
        label_tensor = load_label_to_tensor(label_path)

    return image_tensor, label_tensor


def simple_load_image(image_path, label_path=None, ignore_label=255):

    image_tensor, label_tensor = load_image_tensor_from_path(image_path, label_path)
    image_tensor = tf.expand_dims(tf.cast(image_tensor, tf.float32), axis=0)  # [1, H, W, 3]

    if label_tensor is not None:
        label_tensor = tf.expand_dims(label_tensor, axis=0) # [1, H, W, 1]

    return simple_process_image(image_tensor, label_tensor, ignore_label=ignore_label)


def simple_process_image (image_tensor, label_tensor=None, ignore_label=255):

    image_size = tf.shape(image_tensor)[1:3]

    pad_height = tf.cast(tf.math.ceil(image_size[0] / 32) * 32, tf.int32)
    pad_width = tf.cast(tf.math.ceil(image_size[1] / 32) * 32, tf.int32)

    # pad_height = pad_height if pad_height % 2 != 0 else pad_height + 1
    # pad_width = pad_height if pad_width % 2 != 0 else pad_width + 1

    zero = tf.constant(0, pad_height.dtype)

    r_height = tf.math.floormod(pad_height, 2, name="height_mod")
    r_width = tf.math.floormod(pad_width, 2, name="width_mod")

    cond_height= tf.math.equal(r_height, zero, name="cond_height")
    cond_width = tf.math.equal(r_width, zero, name="cond_width")

    # pad_height = tf.cond(cond_height, lambda:pad_height, lambda:pad_height + 1, name="odd_pad_height")
    # pad_width = tf.cond(cond_width, lambda:pad_width, lambda:pad_width + 1, name="odd_pad_width")

    pad_height += tf.cast(cond_height, tf.int32) 
    pad_width += tf.cast(cond_width, tf.int32)

    pad_height = tf.identity(pad_height, name="target_height")
    pad_width = tf.identity(pad_width, name="target_width")

    from iseg.data_process.utils import pad_to_bounding_box, normalize_value_range

    pad_image_tensor = pad_to_bounding_box(image_tensor, 0, 0, pad_height, pad_width, pad_value=[127.5, 127.5, 127.5])
    pad_image_tensor = normalize_value_range(pad_image_tensor)

    if label_tensor is not None:
        label_tensor = pad_to_bounding_box(label_tensor, 0, 0, pad_height, pad_width, pad_value=ignore_label)

    return pad_image_tensor, label_tensor, image_size

