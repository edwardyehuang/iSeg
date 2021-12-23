# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


class DataAugmentationBase(object):
    def __init__(self, name=None):
        super(DataAugmentationBase, self).__init__()

        if name is None:
            name = type(self).__name__

        self.name = name

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):

        return None


def random_execute_helper(execute_prob, fn0, fn1):

    execute_prob = tf.constant(execute_prob)

    prob = tf.random.uniform((), minval=0, maxval=1.0)

    return tf.cond((execute_prob == 1.0) | (prob <= execute_prob), fn0, fn1)
