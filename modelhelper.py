# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import time

import keras
import tensorflow as tf


from iseg.utils.keras_ops import set_bn_epsilon, set_bn_momentum, set_weight_decay
from iseg.utils.keras3_utils import is_keras3
from iseg.saver.h5_saver import load_h5_weight_by_name

K_GENERAL_POSTFIX = "ckpt.general.h5" # Compatible with keras 2 and 3
K3_POSTFIX = "ckpt.weights.h5"




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
        max_to_keep=20,
        force_use_keras2=False,
    ):
        # Note that, we need to support both keras 2 and 3 in the same class to easier convert in tf keras 2.5

        self.model : keras.Model = model
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.force_use_keras2 = force_use_keras2

        if not is_keras3() or self.force_use_keras2:
            self.ckpt = tf.train.Checkpoint(model=self.model)

            if checkpoint_dir is not None:
                self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=max_to_keep)
            else:
                self.ckpt_manager = None


    def set_optimizer(self, optimizer):

        self.__optimizer = optimizer

    @property
    def optimizer(self):

        if self.__optimizer is None:
            raise ValueError("The optimizer is None")

        return self.__optimizer


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

        if is_keras3() and not self.force_use_keras2:
            return self.restore_checkpoint_keras3()
        else:
            return self.restore_checkpoint_keras2()
    

    def save_checkpoint(self):

        if is_keras3() and not self.force_use_keras2:
            return self.save_checkpoint_keras3()
        else:
            return self.save_checkpoint_keras2()
    


    # code for keras 2

    def restore_checkpoint_keras2(self):

        if self.ckpt_manager is None:
            return None

        last_checkpoint = self.ckpt_manager.latest_checkpoint

        if last_checkpoint is not None:
            self.ckpt.restore(last_checkpoint)

        return last_checkpoint
    

    def save_checkpoint_keras2(self):
        
        return self.ckpt_manager.save()
    

    # code for keras 3

    def restore_checkpoint_keras3(self):
        
        latest_checkpoint = self.get_lastest_checkpoint_keras3()

        if latest_checkpoint is None:
            return None
        
        full_path = tf.io.gfile.join(self.checkpoint_dir, latest_checkpoint)

        latest_checkpoint_postfix = ".".join(latest_checkpoint.split('.')[-3:])

        if latest_checkpoint_postfix == K_GENERAL_POSTFIX:
            print(f"Loading checkpoint {full_path} in general keras format, compatible with keras 2 and 3")
            load_h5_weight_by_name(self.model, full_path)
        elif latest_checkpoint_postfix == K3_POSTFIX:
            print(f"Loading checkpoint {full_path} in keras 3 format")
            self.model.load_weights(full_path)
        else:
            raise ValueError(f"Unknown checkpoint format: {latest_checkpoint_postfix}, path: {full_path}")

        # load_h5_weight_by_name(self.model, full_path)

        return latest_checkpoint
    

    def save_checkpoint_keras3(self, checkpoint_dir=None):

        checkpoint_dir = self.checkpoint_dir if checkpoint_dir is None else checkpoint_dir

        if checkpoint_dir is None:
            print("No checkpoint dir specified, skip saving checkpoint")
            return None

        if not tf.io.gfile.exists(checkpoint_dir):
            tf.io.gfile.makedirs(checkpoint_dir)

        timestr = time.strftime("%Y%m%d-%H%M%S")

        postfix = K3_POSTFIX

        if not is_keras3():
            # Although Keras 2.15 supports saving in keras 3 format, 
            # but the saved file cannot be loaded in keras 3, so we
            # use the general postfix for keras 2 and legacy .h5 format
            postfix = K_GENERAL_POSTFIX
            print("Saving in legacy keras .h5 format, compatible with keras 2 and 3")

        filename = f"id-{timestr}.{postfix}"

        full_path = tf.io.gfile.join(checkpoint_dir, filename)

        self.model.save_weights(full_path)

        print(f"Saved checkpoint to {full_path}")

        # remove old checkpoints
        files = self.get_checkpoint_list_keras3(checkpoint_dir)

        if len(files) > self.max_to_keep:
            files = sorted(files)

            num_to_remove = len(files) - self.max_to_keep

            for i in range(num_to_remove):
                f = files[i]
                full_path = tf.io.gfile.join(checkpoint_dir, f)
                print(f"Removing old checkpoint file: {full_path}")
                tf.io.gfile.remove(full_path)

        return filename


    

    def get_checkpoint_list_keras3(self, checkpoint_dir=None):

        checkpoint_dir = self.checkpoint_dir if checkpoint_dir is None else checkpoint_dir

        if checkpoint_dir is None or not tf.io.gfile.exists(checkpoint_dir):
            print("No checkpoint dir specified or not exists")
            return []
        
        files = tf.io.gfile.listdir(checkpoint_dir)

        ckpt_files = [f for f in files if f.endswith(K3_POSTFIX)]

        # we also need to consider the general postfix for keras 2(legacy .h5 format)
        ckpt_files += [f for f in files if f.endswith(K_GENERAL_POSTFIX)]

        print(f"Found checkpoint files: {ckpt_files}")

        return ckpt_files
    

    def get_lastest_checkpoint_keras3(self, checkpoint_dir=None):

        files = self.get_checkpoint_list_keras3(checkpoint_dir)

        if len(files) == 0:
            print("No checkpoint found")
            return None

        files = sorted(files)

        result = files[-1]

        print(f"Latest checkpoint file: {result}")

        return result
    


