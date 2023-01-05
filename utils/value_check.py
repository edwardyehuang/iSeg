import tensorflow as tf

__check_level = -1

def check_numerics(tensor, message, level=1, name=None):

    x = tensor

    if __check_level < 0 or __check_level >= level:
        x = tf.debugging.check_numerics(
            tensor=tensor,
            message=message,
            name=name,
        )

    return x


def set_check_numerics_level(level):
    __check_level = level
