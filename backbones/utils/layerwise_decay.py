# ===================================================================
# MIT License
# Copyright (c) 2023 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf


def decay_layers_lr (layers=[], rate=0.99):

    current_rate = rate

    for layer in layers:

        print(f"decay lr for {layer.name} with rate = {current_rate}")

        var_list = layer.trainable_weights

        for v in var_list:
            current_lr_multiplier = 1.

            if hasattr(v, 'lr_multiplier'):
                current_lr_multiplier = v.lr_multiplier

            v.lr_multiplier = tf.multiply(
                current_lr_multiplier, 
                current_rate
            )

        current_rate *= rate