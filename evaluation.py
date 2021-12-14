# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from tqdm import tqdm

from iseg.metrics.mean_iou import MeanIOU

from iseg.core_model import SegBase

from iseg.losses.catecrossentropy_ignore_label import catecrossentropy_ignore_label_loss
from iseg.metrics.seg_metric_wrapper import SegMetricWrapper

from common_flags import FLAGS


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
):

    if not isinstance(model, SegBase):
        raise ValueError("ALl model must based on SegBase")

    ds = prepare_dataset(distribute_strategy, data, batch_size, val_image_count=val_image_count)

    processed_count = 0

    print("Evaluate ap : ")

    with distribute_strategy.scope():

        loss_func = catecrossentropy_ignore_label_loss(
            num_class=num_class, ignore_label=ignore_label, batch_size=batch_size, reduction=False
        )

        loss_metrics = tf.keras.metrics.Mean("loss")

        iou_metrics = MeanIOU(num_class)

        iou_metrics = SegMetricWrapper(iou_metrics, num_class=num_class, ignore_label=ignore_label, name="IOU")

        print("Current image format = {}".format(tf.keras.backend.image_data_format()))

        with tqdm(total=val_image_count) as pbar:
            for inputs in ds:
                eval_step(inputs, model, scale_rates, flip, loss_func, loss_metrics, [iou_metrics], distribute_strategy)

                processed_count += batch_size

                pbar.update(batch_size)
                pbar.set_description(
                    "Processed : {:}, current loss = {:4f}, current IOU = {:.2f} %".format(
                        processed_count, loss_metrics.result(), iou_metrics.result().numpy() * 100
                    )
                )

    mean_loss = loss_metrics.result()
    loss_metrics.reset_states()

    mean_iou = iou_metrics.result()
    iou_metrics.reset_states()

    print("Mean loss on val set :", mean_loss.numpy())
    print("Mean iou on val set :", mean_iou.numpy())

    return mean_iou


@tf.function
def eval_step(ds_inputs, model: SegBase, scale_rates, flip, loss_func, loss_metrics, mertics, distribute_strategy):
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

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    ds = data
    # ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=False)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    ds = distribute_strategy.experimental_distribute_dataset(ds)

    return ds
