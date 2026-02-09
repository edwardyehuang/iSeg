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

from iseg.utils.model_utils import create_compiled_model



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

    with distribute_strategy.scope():
        model : keras.Model = create_compiled_model(
            model=model,
            num_class=num_class,
            ignore_label=ignore_label,
            batch_size=batch_size,
            jit_compile=False,
        )

    print(f"Total eval steps = Total image count {val_image_count} // batch size {batch_size} = {val_image_count // batch_size}")

    model.evaluate(
        ds, 
        batch_size=batch_size, 
        verbose=1, 
        steps=val_image_count // batch_size,
        use_multiprocessing=True,
    )



def prepare_dataset(distribute_strategy, data, batch_size=16, val_image_count=0):

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    ds = data
    # ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds