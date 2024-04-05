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
    pre_compute_fns=[],
):

    if not isinstance(model, SegBase):
        raise ValueError("ALl model must based on SegBase")

    ds = prepare_dataset(distribute_strategy, data, batch_size, val_image_count=val_image_count)

    processed_count = 0

    print("Evaluate mIOU : ")

    with distribute_strategy.scope():

        loss_func = catecrossentropy_ignore_label_loss(
            num_class=num_class, ignore_label=ignore_label, batch_size=batch_size, reduction=False
        )

        loss_metrics = tf.keras.metrics.Mean("loss")

        iou_metrics = MeanIOU(num_class)

        iou_metrics = SegMetricWrapper(iou_metrics, num_class=num_class, ignore_label=ignore_label, name="IOU")

        if pre_compute_fns is not None and isinstance(pre_compute_fns, list):
            for fn in pre_compute_fns:
                iou_metrics.add_pre_compute_fn(fn)

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
    mean_iou = iou_metrics.result()
    
    print("-----------------------------------------------")

    print(f"Mean loss on val set : {mean_loss.numpy()}")
    print(f"Mean IoU on val set : {mean_iou.numpy()}")

    print("-----------------------------------------------")

    print(f"Per-class IoU on val set :")

    pre_class_iou =  iou_metrics.metric.per_class_result()

    print(pre_class_iou)

    loss_metrics.reset_states()
    iou_metrics.reset_states()

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
