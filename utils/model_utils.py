# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import keras
import numpy as np
import math


from iseg.metrics.mean_iou import MeanIOU
from iseg.metrics.seg_metric_wrapper import SegMetricWrapper
from iseg.losses.catecrossentropy_ignore_label import catecrossentropy_ignore_label_loss
from iseg.core_model import SegFoundation

from iseg.utils.keras_ops import capture_func
from iseg.utils.train_utils import exclude_no_weight_decay_layers_in_optimizer

def create_compiled_model(
    model:SegFoundation,
    num_class, 
    ignore_label=255,
    class_weights=None, 
    batch_size=1, 
    epoch_steps=1000, 
    initial_epoch=0,
    jit_compile=None,
    optimizer:keras.optimizers.Optimizer=None,
):

    assert isinstance(model, SegFoundation), "Current only support SegFoundation based model"

    # Loss functions

    losses_func = getattr(model, "custom_losses", None)

    if losses_func is None or not callable(losses_func):
        losses_func = catecrossentropy_ignore_label_loss

    losses = losses_func(
        num_class=num_class, 
        ignore_label=ignore_label,
        class_weights=class_weights,
        batch_size=batch_size, 
        reduction=False)

    # Loss weights:

    losses_weights = None
    losses_weights_func = capture_func(model, "custom_losses_weights")

    if losses_weights_func is not None:
        losses_weights = losses_weights_func()

    # Metrics

    metrics = None

    metrics_func = getattr(model, "custom_metrics", None)

    if metrics_func is None or not callable(metrics_func):
        metrics_func = __get_default_metrics

    metrics = metrics_func(num_class, ignore_label)

    # Handle no weight decay layers
    exclude_no_weight_decay_layers_in_optimizer(
        optimizer=optimizer,
        model=model
    )

    # Compile

    model.compile(
        optimizer=optimizer, 
        metrics=metrics, 
        loss=losses, 
        loss_weights=losses_weights,
        jit_compile=jit_compile,
    )

    if initial_epoch != -1:
        model.optimizer.iterations.assign(epoch_steps * initial_epoch)

    return model



def __get_default_metrics(num_class, ignore_label):

    iou_metrics = MeanIOU(num_class)
    iou_metrics = SegMetricWrapper(iou_metrics, num_class=num_class, ignore_label=ignore_label, name="IOU")

    return [iou_metrics]