from distutils.version import LooseVersion

import tensorflow as tf
import keras

def is_keras3():
    if LooseVersion(tf.version.VERSION) < LooseVersion("2.15.0"):
        return False
    
    keras_version = LooseVersion(keras.__version__)

    if keras_version < LooseVersion("3.0.0"):
        return False
    
    if LooseVersion(keras.__version__) < LooseVersion("3.0.5"):
        raise ValueError(f"Keras {keras.__version__} is not supported, please use Keras 3.0.5 or later.")
    
    return True


def is_keras2_15():

    if LooseVersion(tf.version.VERSION) < LooseVersion("2.15.0"):
        return False
    
    keras_version = LooseVersion(keras.__version__)

    return keras_version == LooseVersion("2.15.0")


def print_keras_version():
    print(f"Keras version: {keras.__version__}")