import tensorflow as tf

from PIL import Image

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



    