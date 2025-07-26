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

KERAS_SCALE_FUNC = keras.layers.Rescaling(
    scale=1.0 / 255.0,
)

KERAS_NORM_SCALE_FUNC = keras.layers.Normalization(
    mean=[0.485, 0.456, 0.406],
    variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2],
)

KERAS_NORM_SCALE_FUNC_INVERT = keras.layers.Normalization(
    mean=[0.485, 0.456, 0.406],
    variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2],
    invert=True,
)

def preprocess_zero_mean_unit_range(inputs):
    """Map image values from [0, 255] to [-1, 1]."""

    preprocessed_inputs = (2.0 / 255.0) * tf.cast(inputs, "float32") - 1.0

    return tf.cast(preprocessed_inputs, dtype=inputs.dtype)


@tf.autograph.experimental.do_not_convert
def keras_norm_preprocess(inputs, scale=False):

    x = inputs

    if scale:
        norm_func = KERAS_NORM_SCALE_FUNC

        if not KERAS_SCALE_FUNC.built:
            KERAS_SCALE_FUNC.build((None, None, 3))
            
        x = KERAS_SCALE_FUNC(x)
    else:
        norm_func = KERAS_NORM_FUNC

    if not norm_func.built:
        norm_func.build((None, None, 3))

    x = norm_func(x)

    return tf.cast(x, dtype=inputs.dtype)


@tf.autograph.experimental.do_not_convert
def keras_norm_preprocess_invert(inputs, scale=False):

    x = inputs

    if scale:
        norm_func = KERAS_NORM_SCALE_FUNC_INVERT
    else:
        norm_func = KERAS_NORM_FUNC_INVERT

    if not norm_func.built:
        norm_func.build((None, None, 3))

    x = norm_func(x)

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