from distutils.version import LooseVersion

import tensorflow as tf

if LooseVersion(tf.version.VERSION) < LooseVersion("2.15.0"):
    from tensorflow import keras
else:
    import keras

REPLACE_SLASH = False

from iseg.utils.slash_utils import replace_slash

def _N(name):
    return replace_slash(name)


def is_keras3():
    if LooseVersion(tf.version.VERSION) < LooseVersion("2.15.0"):
        return False
    
    return LooseVersion(keras.version()) >= LooseVersion("3.0.0")


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

        super().__init__(trainable, name, dtype, dynamic, **kwargs)