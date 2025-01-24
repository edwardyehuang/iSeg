from iseg.data_process.input_norm_types import InputNormTypes


def get_mean_pixel(input_norm_types):
    if input_norm_types == InputNormTypes.KERAS:
        return [123.675, 116.28, 103.53]

    return [127.5, 127.5, 127.5]