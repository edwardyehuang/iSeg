# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.utils.keras_ops import set_bn_epsilon, set_bn_momentum, set_weight_decay


def model_common_setup(
    model,
    restore_checkpoint=True,
    checkpoint_dir=None,
    max_checkpoints_to_keep=1,
    weight_decay=None,
    decay_norm_vars=False,
    bn_epsilon=None,
    bn_momentum=None,
    backbone_bn_momentum=None,
    inference_sliding_window_size=None,
):

    model.inference_sliding_window_size = inference_sliding_window_size

    model_helper = ModelHelper(model, checkpoint_dir, max_checkpoints_to_keep)

    if restore_checkpoint:
        model_helper.restore_checkpoint()

    if weight_decay is not None:
        set_weight_decay(model_helper.model, weight_decay, decay_norm_vars)

    if bn_epsilon is not None:
        set_bn_epsilon(model_helper.model, bn_epsilon)

    if bn_momentum is not None:
        set_bn_momentum(model_helper.model, bn_momentum)

    if backbone_bn_momentum is not None and hasattr(model_helper.model, "backbone"):
        set_bn_momentum(model_helper.model.backbone, backbone_bn_momentum)

    # frezze_batch_norms(model_helper.model, FLAGS.bn_freeze)

    return model_helper


class ModelHelper:
    def __init__(self, model: tf.keras.Model, checkpoint_dir, max_to_keep=20):

        self.model = model

        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=max_to_keep)

    def set_optimizer(self, optimizer):

        self.__optimizer = optimizer

    @property
    def optimizer(self):

        if self.__optimizer is None:
            raise ValueError("The optimizer is None")

        return self.__optimizer

    def restore_checkpoint(self):

        last_checkpoint = self.ckpt_manager.latest_checkpoint

        if last_checkpoint != None:
            self.ckpt.restore(last_checkpoint).expect_partial()

        return last_checkpoint

    def save_checkpoint(self):

        return self.ckpt_manager.save()

    def list_latest_ckpt_vars(self):
        last_checkpoint = self.ckpt_manager.latest_checkpoint

        return tf.train.list_variables(last_checkpoint)
