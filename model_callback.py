# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.utils.keras_ops import capture_func


class ModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, binding_model):

        super(ModelCallback, self).__init__()

        self.binding_model = binding_model

        self.model_epoch_begin_func = capture_func(self.binding_model, "on_epoch_begin")

        self.model_epoch_end_func = capture_func(self.binding_model, "on_epoch_end")

        self.model_test_batch_end_func = capture_func(self.binding_model, "on_test_batch_end")

    def on_epoch_begin(self, epoch, logs={}):
        if self.model_epoch_begin_func is not None:
            self.model_epoch_begin_func(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):

        if self.model_epoch_end_func is not None:
            self.model_epoch_end_func(epoch, logs)

    def on_test_batch_end(self, batch, logs={}):

        if self.model_test_batch_end_func is not None:
            self.model_test_batch_end_func(batch, logs)
