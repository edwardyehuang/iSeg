# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
from iseg.layers.syncbn import SyncBatchNormalization
from iseg.layers.groupnorm import GroupNormalization

GLOBAL = "global"
BATCH_NORM = "batch_norm"
SYNC_BATCH_NORM = "sync_batch_norm"
GROUP_NROM = "group_norm"


def global_norm_method():

    return SYNC_BATCH_NORM


def normalization(
    axis: int = -1,
    momentum=0.9,
    epsilon: float = 1e-3,
    center: bool = True,
    scale: bool = True,
    beta_initializer="zeros",
    gamma_initializer="ones",
    moving_mean_initializer="zeros",
    moving_variance_initializer="ones",
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
    groups=16,
    method=GLOBAL,
    trainable=True,
    name: str = None,
    **kwargs
):

    if method == GLOBAL or method == None:
        method = global_norm_method()

    if method == BATCH_NORM:
        name = name if name is not None else BATCH_NORM
        return tf.keras.layers.BatchNormalization(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            trainable=trainable,
            name=name,
            **kwargs
        )

    elif method == SYNC_BATCH_NORM:
        name = name if name is not None else BATCH_NORM
        return SyncBatchNormalization(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            trainable=trainable,
            name=name,
            **kwargs
        )

    elif method == GROUP_NROM:
        name = name if name is not None else GROUP_NROM
        return GroupNormalization(
            groups=groups,
            axis=axis,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            trainable=trainable,
            name=name,
            **kwargs
        )
    else:
        raise ValueError("Not support norm mathod = {}".format(method))
