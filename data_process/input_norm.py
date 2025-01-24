import keras

from iseg.data_process.input_norm_types import InputNormTypes


def preprocess_zero_mean_unit_range(inputs):
    """Map image values from [0, 255] to [-1, 1]."""

    preprocessed_inputs = (2.0 / 255.0) * keras.backend.cast(inputs, "float32") - 1.0

    return keras.backend.cast(preprocessed_inputs, dtype=inputs.dtype)


def keras_norm_preprocess(inputs):

    x = inputs
    x = keras.layers.Normalization(
        mean=[123.675, 116.28, 103.53],
        variance=[58.395 ** 2, 57.12 ** 2, 57.375 ** 2],
    )(x)

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
    

def normalize_input_value_range_by_model(
    image,
    model,
):
    if not hasattr(model, "input_norm_type"):
        raise ValueError("model must have input_norm_type attribute")
    
    return normalize_input_value_range(image, input_norm_type=model.input_norm_type)