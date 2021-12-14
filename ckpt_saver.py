# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.modelhelper import ModelHelper


class CheckpointSaver(tf.keras.callbacks.Callback):
    def __init__(self, model_helper: ModelHelper):

        super(CheckpointSaver, self).__init__()

        self.model_helper = model_helper

    def on_epoch_end(self, epoch, logs={}):
        self._save_ckpt()

    def _save_ckpt(self):

        path = self.model_helper.save_checkpoint()
        print()
        print("Checkpoint is saved to ", path)
        print()
