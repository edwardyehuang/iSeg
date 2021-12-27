import tensorflow as tf
import sys


def PlaceHolder(*args, **kwargs):

    return sys.modules["placeholder_func"](*args, **kwargs)
