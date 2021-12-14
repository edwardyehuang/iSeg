# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.backbones.resnet_common import *
from iseg.utils.keras_ops import HookLayer


def hook_aux_loss_layer(resnet_model):

    model: ResNet = resnet_model

    target_block = model.stacks[-1]
    target_block.block1 = HookLayer(target_block.block1)
    result = lambda: target_block.block1.input_features

    return result
