import tensorflow as tf

from PIL import Image

from iseg.data_process.augments.pad_to_odd_augment import pad_to_odd
from iseg.data_process.input_norm_types import InputNormTypes
from iseg.data_process.input_norm import normalize_input_value_range
from iseg.data_process.mean_pixel import get_mean_pixel


def load_label_to_tensor(label_path):
    label_tensor = tf.py_function(_load_label_to_tensor_internel, [label_path], tf.int32)
    label_tensor.set_shape([None, None, 1])

    return label_tensor


def _load_label_to_tensor_internel(path_tensor):

    if isinstance(path_tensor, str):
        label_path = path_tensor
    else:
        label_path = path_tensor.numpy()

    label_image = Image.open(label_path)
    label_array = tf.keras.preprocessing.image.img_to_array(label_image, "channels_last")
    label_image.close()


    label_tensor = tf.convert_to_tensor(label_array)
    label_tensor = tf.cast(label_tensor, tf.int32)

    return label_tensor


def load_image_tensor_from_path(image_path, label_path=None):

    image_data = tf.io.read_file(image_path)

    try:
        image_tensor = tf.image.decode_jpeg(
            image_data, 
            channels=3, 
            dct_method="INTEGER_ACCURATE", 
            try_recover_truncated=True
        )

        image_tensor = tf.cast(image_tensor, tf.float32)

        label_tensor = None

        if label_path is not None:
            label_tensor = load_label_to_tensor(label_path)

        return image_tensor, label_tensor

    except:

        if label_path is not None:
            raise ValueError(f"Error: {image_path} or {label_path} is not a valid image.")
        else:
            raise ValueError(f"Error: {image_path} is not a valid JPEG image.")


def simple_load_image(
    image_path, 
    label_path=None, 
    ignore_label=255,
    fit_downsample_rate=32,
    pad_to_odd_shape=False,
    input_norm_type=InputNormTypes.ZERO_MEAN,
):

    image_tensor, label_tensor = load_image_tensor_from_path(image_path, label_path)
    image_tensor = tf.expand_dims(tf.cast(image_tensor, tf.float32), axis=0)  # [1, H, W, 3]

    if label_tensor is not None:
        label_tensor = tf.expand_dims(label_tensor, axis=0) # [1, H, W, 1]

    return simple_process_image(
        image_tensor, 
        label_tensor, 
        ignore_label=ignore_label,
        fit_downsample_rate=fit_downsample_rate,
        pad_to_odd_shape=pad_to_odd_shape,
        input_norm_type=input_norm_type,
    )


def get_simple_process_padding_size(
    x,
    fit_downsample_rate=32,
    pad_to_odd_shape=True,
):
    
    image_size = tf.shape(x)[1:3]

    pad_height = tf.cast(tf.math.ceil(image_size[0] / fit_downsample_rate) * fit_downsample_rate, tf.int32)
    pad_width = tf.cast(tf.math.ceil(image_size[1] / fit_downsample_rate) * fit_downsample_rate, tf.int32)

    if pad_to_odd_shape:
        zero = tf.constant(0, pad_height.dtype)

        r_height = tf.math.floormod(pad_height, 2, name="height_mod")
        r_width = tf.math.floormod(pad_width, 2, name="width_mod")

        cond_height= tf.math.equal(r_height, zero, name="cond_height")
        cond_width = tf.math.equal(r_width, zero, name="cond_width")

        pad_height += tf.cast(cond_height, tf.int32) 
        pad_width += tf.cast(cond_width, tf.int32)

    pad_height = tf.identity(pad_height, name="target_height")
    pad_width = tf.identity(pad_width, name="target_width")

    return pad_height, pad_width


def simple_process_image (
    image_tensor, 
    label_tensor=None, 
    ignore_label=255,
    fit_downsample_rate=32,
    pad_to_odd_shape=True,
    input_norm_type=InputNormTypes.ZERO_MEAN,
):

    image_size = tf.shape(image_tensor)[1:3]

    pad_height, pad_width = get_simple_process_padding_size(
        image_tensor,
        fit_downsample_rate=fit_downsample_rate,
        pad_to_odd_shape=pad_to_odd_shape,
    )

    from iseg.data_process.utils import pad_to_bounding_box

    mean_pixel = get_mean_pixel(input_norm_type)

    pad_image_tensor = pad_to_bounding_box(image_tensor, 0, 0, pad_height, pad_width, pad_value=mean_pixel)
    pad_image_tensor = normalize_input_value_range(pad_image_tensor, input_norm_type=input_norm_type)

    if label_tensor is not None:
        label_tensor = pad_to_bounding_box(label_tensor, 0, 0, pad_height, pad_width, pad_value=ignore_label)

    return pad_image_tensor, label_tensor, image_size

