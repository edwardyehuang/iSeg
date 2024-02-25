import tensorflow as tf

REPLACE_SLASH = False

from iseg.utils.slash_utils import replace_slash

class Keras3_Model_Wrapper(tf.keras.Model):

    def __init__(self, *args, name=None, **kwargs):

        name = replace_slash(name)

        super().__init__(*args, name=name, **kwargs)


class Keras3_Layer_Wrapper(tf.keras.layers.Layer):

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