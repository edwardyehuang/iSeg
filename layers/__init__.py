# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import functools
from distutils.version import LooseVersion

import tensorflow as tf

from iseg.layers.normalizations import normalization, BATCH_NORM, SYNC_BATCH_NORM, GROUP_NROM

if LooseVersion(tf.version.VERSION) < LooseVersion("2.11.0"):
    from iseg.layers.syncbn import SyncBatchNormalization
elif LooseVersion(tf.version.VERSION) < LooseVersion("2.14.0"):
    SyncBatchNormalization = tf.keras.layers.experimental.SyncBatchNormalization
else:
    SyncBatchNormalization = functools.partial(
        tf.keras.layers.BatchNormalization, 
        synchronized=True
    )

from iseg.layers.dense_ext import DenseExt
from iseg.layers.model_builder import ConvBnRelu
