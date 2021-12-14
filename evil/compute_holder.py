# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

# WIP


class SimpleHolder(object):
    def __init__(self, distribute_strategy) -> None:
        super().__init__()

        self.distribute_strategy = distribute_strategy

    @tf.function
    def call(self, inputs):

        a = inputs * tf.random.uniform(tf.shape(inputs))

        return

    def data_generator(self):

        return tf.random.uniform(shape=(16, 32, 32, 3))

    def __call__(self, inputs):

        with self.distribute_strategy.scope():

            self.distribute_strategy.run(self.call, args=(inputs,))
