import tensorflow as tf
import keras

from iseg.data_process.input_norm_types import InputNormTypes


KERAS_NORM_FUNC = keras.layers.Normalization(
    mean=[123.675, 116.28, 103.53],
    variance=[58.395 ** 2, 57.12 ** 2, 57.375 ** 2],
)

KERAS_NORM_FUNC_INVERT = keras.layers.Normalization(
    mean=[123.675, 116.28, 103.53],
    variance=[58.395 ** 2, 57.12 ** 2, 57.375 ** 2],
    invert=True,
)

def preprocess_zero_mean_unit_range(inputs):
    """Map image values from [0, 255] to [-1, 1]."""

    preprocessed_inputs = (2.0 / 255.0) * tf.cast(inputs, "float32") - 1.0

    return tf.cast(preprocessed_inputs, dtype=inputs.dtype)


@tf.autograph.experimental.do_not_convert
def keras_norm_preprocess(inputs):

    x = inputs

    if not KERAS_NORM_FUNC.built:
        KERAS_NORM_FUNC.build((None, None, 3))

    x = KERAS_NORM_FUNC(x)

    return tf.cast(x, dtype=inputs.dtype)


@tf.autograph.experimental.do_not_convert
def keras_norm_preprocess_invert(inputs):

    x = inputs

    if not KERAS_NORM_FUNC_INVERT.built:
        KERAS_NORM_FUNC_INVERT.build((None, None, 3))

    x = KERAS_NORM_FUNC_INVERT(x)

    return keras.backend.cast(x, dtype=inputs.dtype)


def normalize_input_value_range(
    image, 
    input_norm_type : InputNormTypes = InputNormTypes.ZERO_MEAN
):
    
    if input_norm_type == InputNormTypes.NONE:
        return image
    elif input_norm_type == InputNormTypes.ZERO_MEAN:
        return preprocess_zero_mean_unit_range(image)
    elif input_norm_type == InputNormTypes.KERAS:
        return keras_norm_preprocess(image)
    else:
        raise ValueError(f"Unsupported input_norm_type: {input_norm_type}")
    

def invert_normalize_input_value_range(
    image,
    input_norm_type : InputNormTypes = InputNormTypes.ZERO_MEAN
):
    if input_norm_type == InputNormTypes.NONE:
        return image
    elif input_norm_type == InputNormTypes.ZERO_MEAN:
        raise NotImplementedError("Not implemented")
    elif input_norm_type == InputNormTypes.KERAS:
        return keras_norm_preprocess_invert(image)
    else:
        raise ValueError(f"Unsupported input_norm_type: {input_norm_type}")
    

def normalize_input_value_range_by_model(
    image,
    model,
):
    if not hasattr(model, "input_norm_type"):
        raise ValueError("model must have input_norm_type attribute")
    
    return normalize_input_value_range(image, input_norm_type=model.input_norm_type)