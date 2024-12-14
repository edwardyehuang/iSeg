import os
from distutils.version import LooseVersion

from iseg.utils.version_utils import is_keras3, is_keras2_15

import tensorflow as tf

if LooseVersion(tf.version.VERSION) < LooseVersion("2.15.0"):
    from tensorflow import keras
else:
    import keras

from iseg.utils.slash_utils import replace_slash

def _N(name):
    return replace_slash(name)


def print_keras_version():
    print(f"Keras version: {keras.__version__}")


class Keras3_Model_Wrapper(keras.Model):

    def __init__(self, *args, name=None, **kwargs):

        name = replace_slash(name)

        super().__init__(*args, name=name, **kwargs)


class Keras3_Layer_Wrapper(keras.layers.Layer):

    def __init__(
        self,
        trainable=True, 
        name=None, 
        dtype=None, 
        dynamic=False, 
        **kwargs
    ):
        
        name = replace_slash(name)

        if not is_keras3():
            super().__init__(
                trainable=trainable, 
                name=name, 
                dtype=dtype, 
                dynamic=dynamic, 
                **kwargs
            )
        else:
            super().__init__(
                trainable=trainable, 
                name=name, 
                dtype=dtype, 
                **kwargs
            )