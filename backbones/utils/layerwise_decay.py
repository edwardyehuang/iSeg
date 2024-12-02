# ===================================================================
# MIT License
# Copyright (c) 2023 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf
import numpy as np

from iseg.utils.train_utils import set_weights_lr_multiplier


def decay_layers_lr (layers=[], weights=[], rate=0.99):

    num_layers = len(layers)

    for i in range(num_layers):

        layer = layers[i]
        current_rate = rate ** (num_layers - i - 2)

        if isinstance(layer, tuple):
            layer = list(layer)

        if not isinstance(layer, list):
            layer = [layer]

        for sub_layer in layer:
            print(f"decay lr for {sub_layer.name} with rate = {current_rate}")

            var_list = sub_layer.trainable_weights

            for v in var_list:
                current_lr_multiplier = current_rate

                if hasattr(v, 'lr_multiplier'):
                    current_lr_multiplier *= v.lr_multiplier
                    print(f"Found existing lr_multiplier = {v.lr_multiplier}, set new lr_multiplier = {current_lr_multiplier}")

                set_weights_lr_multiplier(
                    v, 
                    lr_multiplier=current_lr_multiplier
                )

    
    current_rate = rate ** (num_layers - 1)

    for weight in weights:

        current_lr_multiplier = 1.

        if hasattr(weight, 'lr_multiplier'):
            current_lr_multiplier = weight.lr_multiplier

        set_weights_lr_multiplier(
            weight, 
            lr_multiplier=current_lr_multiplier * current_rate
        )