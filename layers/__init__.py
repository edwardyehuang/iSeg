# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import functools
from distutils.version import LooseVersion

import tensorflow as tf

from iseg.layers.normalizations import normalization, BATCH_NORM, SYNC_BATCH_NORM, GROUP_NROM, SyncBatchNormalization
from iseg.layers.dense_ext import DenseExt
from iseg.layers.model_builder import ConvNormAct
