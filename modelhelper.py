# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

class ModelHelper:

    def __init__ (self, model : tf.keras.Model, checkpoint_dir, max_to_keep = 20):
        
        self.model = model

        self.ckpt = tf.train.Checkpoint(model = self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep = max_to_keep)

    def set_optimizer (self, optimizer):

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
    
    
    def list_latest_ckpt_vars (self):
        last_checkpoint = self.ckpt_manager.latest_checkpoint

        return tf.train.list_variables(last_checkpoint)
