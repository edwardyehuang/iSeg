import tensorflow as tf
import keras

from iseg.data_process.input_norm_types import InputNormTypes


def preprocess_zero_mean_unit_range(inputs):
    """Map image values from [0, 255] to [-1, 1]."""

    preprocessed_inputs = (2.0 / 255.0) * tf.cast(inputs, "float32") - 1.0

    return tf.cast(preprocessed_inputs, dtype=inputs.dtype)


@tf.autograph.experimental.do_not_convert
def keras_norm_preprocess(
    inputs, 
    scale=False, 
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375]
):

    x = inputs

    if scale:
        x /= 255.0
        mean = [m / 255.0 for m in mean]
        std = [s / 255.0 for s in std]

    variance = [s ** 2 for s in std]

    x = (x - mean) / tf.maximum(
        tf.sqrt(variance), keras.backend.epsilon()
    )

    return tf.cast(x, dtype=inputs.dtype)


@tf.autograph.experimental.do_not_convert
def keras_norm_preprocess_invert(
    inputs, 
    scale=False,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375]
):

    x = inputs

    if scale:
        mean = [m / 255.0 for m in mean]
        std = [s / 255.0 for s in std]

    variance = [s ** 2 for s in std]

    x = mean + (
        x * tf.maximum(tf.sqrt(variance), keras.backend.epsilon())
    )
    
    if scale:
        x *= 255.0

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
    elif input_norm_type == InputNormTypes.KEARS_SCALE:
        return keras_norm_preprocess(image, scale=True)
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
    elif input_norm_type == InputNormTypes.KEARS_SCALE:
        return keras_norm_preprocess_invert(image, scale=True)
    else:
        raise ValueError(f"Unsupported input_norm_type: {input_norm_type}")
    

def normalize_input_value_range_by_model(
    image,
    model,
):
    if not hasattr(model, "input_norm_type"):
        raise ValueError("model must have input_norm_type attribute")
    
    return normalize_input_value_range(image, input_norm_type=model.input_norm_type)