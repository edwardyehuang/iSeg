# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import keras

from tqdm import tqdm

from iseg.metrics.mean_iou import MeanIOU

from iseg.core_model import SegFoundation

from iseg.losses.catecrossentropy_ignore_label import catecrossentropy_ignore_label_loss
from iseg.metrics.seg_metric_wrapper import SegMetricWrapper



def evaluate(
    distribute_strategy,
    model,
    data,
    batch_size,
    num_class,
    ignore_label=255,
    scale_rates=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    flip=True,
    val_image_count=0,
    pre_compute_fns=[],
):

    if not isinstance(model, SegFoundation):
        raise ValueError("ALl model must based on SegFoundation")

    ds = prepare_dataset(distribute_strategy, data, batch_size, val_image_count=val_image_count)

   



@tf.function(autograph=False)
def eval_step(
    ds_inputs, 
    model: SegFoundation, 
    scale_rates, 
    flip, 
    loss_func, 
    loss_metrics, 
    mertics, 
    distribute_strategy
):
    def step_fn(inputs):
        images, labels = inputs

        predictions = model.inference_with_multi_scales(images, training=False, scale_rates=scale_rates, flip=flip)

        # predictions = model.inference(images, False)

        loss = loss_func(labels, predictions)

        loss_metrics.update_state(loss)

        for mertic in mertics:
            mertic.update_state(labels, predictions)

    return distribute_strategy.run(step_fn, args=(ds_inputs,))


def prepare_dataset(distribute_strategy, data, batch_size=16, val_image_count=0):

    # AUTOTUNE = tf.data.experimental.AUTOTUNE

    ds = data
    # ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)

    # ds = ds.prefetch(buffer_size=AUTOTUNE)

    ds = distribute_strategy.experimental_distribute_dataset(ds)

    return ds