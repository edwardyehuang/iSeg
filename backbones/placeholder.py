import tensorflow as tf

class PlaceHolder (tf.keras.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def call(self, inputs, training = None):

        raise NotImplementedError()