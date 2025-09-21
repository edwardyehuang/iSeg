# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import keras

from iseg.utils.keras_ops import set_bn_epsilon, set_bn_momentum, set_weight_decay

from iseg.utils.keras3_utils import is_keras3


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

    if weight_decay is not None and weight_decay > 0:
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
    def __init__(
        self, 
        model: keras.Model, 
        checkpoint_dir,
        max_to_keep=20
    ):

        self.model = model

        self.ckpt = tf.train.Checkpoint(model=self.model)

        if checkpoint_dir is not None:
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=max_to_keep)
        else:
            self.ckpt_manager = None


    def print_trackable_variables(self, items=[]):
        
        for item in items:

            if hasattr(item, 'path'):
                print(f"path = {item.path}")

            if hasattr(item, 'name'):
                print(f"name = {item.name}")

            sub_items = tf.train.TrackableView(item).children(item).items()
            sub_items = dict(sub_items)
            sub_items = list(sub_items.values())

            self.print_trackable_variables(sub_items)


    def restore_checkpoint(self):

        # print trackable variables
        # print("Trackable variables:")
        # self.print_trackable_variables([self.model])

        if is_keras3(): # TODO: Implement restoring for Keras 3
            raise NotImplementedError("Restoring checkpoints is not implemented for Keras 3 yet")
        else:
            return self.restore_checkpoint_keras2()
    

    def restore_checkpoint_keras2(self):

        if self.ckpt_manager is None:
            return None

        last_checkpoint = self.ckpt_manager.latest_checkpoint

        if last_checkpoint is not None:
            self.ckpt.restore(last_checkpoint)

        return last_checkpoint


    def save_checkpoint(self):

        if is_keras3():# TODO: Implement saving for Keras 3
            raise NotImplementedError("Saving checkpoints is not implemented for Keras 3 yet")
        else:
            return self.save_checkpoint_keras2()
    

    def save_checkpoint_keras2(self):
        
        return self.ckpt_manager.save()


    def list_latest_ckpt_vars(self):

        if self.ckpt_manager is None:
            return None

        last_checkpoint = self.ckpt_manager.latest_checkpoint

        return tf.train.list_variables(last_checkpoint)
