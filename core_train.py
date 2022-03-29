# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import numpy as np
import math


from iseg.metrics.mean_iou import MeanIOU
from iseg.metrics.seg_metric_wrapper import SegMetricWrapper
from iseg.losses.catecrossentropy_ignore_label import catecrossentropy_ignore_label_loss
from iseg.ckpt_saver import CheckpointSaver
from iseg.model_callback import ModelCallback
from iseg.core_model import SegFoundation

from iseg.optimizers.multi_optimizer import MultiOptimizer

from iseg.utils.keras_ops import capture_func


class CoreTrain(object):
    def __init__(self, model_helper, train_dataset, val_dataset=None, val_image_count=0, use_tpu=False):

        self.model_helper = model_helper
        self.training_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_image_count = val_image_count

        self.use_tpu = use_tpu

    def create_trainable_model(
        self, 
        num_class, 
        ignore_label=255,
        class_weights=None, 
        batch_size=1, 
        epoch_steps=1000, 
        initial_epoch=0):

        model = self.model_helper.model

        assert isinstance(model, SegFoundation), "Current only support SegFoundation based model"

        model : SegFoundation = model

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
            metrics_func = self.__get_default_metrics

        metrics = metrics_func(num_class, ignore_label)

        # Multi LR

        optimizer = self.model_helper.optimizer

        if isinstance(optimizer, list):
            optimizer = self.handle_multiple_optimizers(model, optimizer)

        # Compile

        model.compile(optimizer=optimizer, metrics=metrics, loss=losses, loss_weights=losses_weights)

        if initial_epoch != -1:
            model.optimizer.iterations.assign(epoch_steps * initial_epoch)

        return model

    def train(
        self,
        distribute_strategy,
        num_class=21,
        ignore_label=255,
        class_weights=None,
        batch_size=1,
        eval_batch_size=None,
        shuffle_rate=100,
        epoch_steps=1000,
        initial_epoch=0,
        train_epoches=30,
        tensorboard_dir="tensorboard",
        verbose=1,
    ):

        if eval_batch_size is None:
            eval_batch_size = batch_size

        with distribute_strategy.scope():
            model = self.create_trainable_model(
                num_class,
                ignore_label=ignore_label,
                class_weights=class_weights,
                batch_size=batch_size,
                epoch_steps=epoch_steps,
                initial_epoch=initial_epoch,
            )

        if initial_epoch == -1:
            current_iter = model.optimizer.iterations.value()
            initial_epoch = current_iter // epoch_steps

        train_ds = self.prepare_train_dataset(model, batch_size, shuffle_rate)
        eval_ds = self.prepare_val_dataset(model, eval_batch_size)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir, histogram_freq=0, write_images=False
        )
        checkpoint_saver = CheckpointSaver(self.model_helper)
        model_callback = ModelCallback(self.model_helper.model)

        val_steps = None if eval_ds is None else int(math.ceil(self.val_image_count / eval_batch_size))

        # Note, we do not apply the shuffle in keras.model.fit as it has already shuffled in tf.data
        model.fit(
            train_ds,
            epochs=train_epoches,
            validation_data=eval_ds,
            shuffle=False,
            callbacks=[tensorboard_callback, checkpoint_saver, model_callback],
            initial_epoch=initial_epoch,
            steps_per_epoch=epoch_steps,
            validation_steps=val_steps,
            verbose=verbose,
        )

    def __get_default_metrics(self, num_class, ignore_label):

        iou_metrics = MeanIOU(num_class)
        iou_metrics = SegMetricWrapper(iou_metrics, num_class=num_class, ignore_label=ignore_label, name="IOU")

        return [iou_metrics]

    def prepare_train_dataset(self, model, batch_size=1, shuffle_rate=100):

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = self.handle_custom_dataprocess(self.training_dataset, model)
        ds = ds.shuffle(shuffle_rate)
        ds = ds.repeat()
        ds = ds.batch(batch_size, drop_remainder=self.use_tpu)

        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def prepare_val_dataset(self, model, batch_size=1):

        if self.val_dataset is None:
            return None

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = self.handle_custom_dataprocess(self.val_dataset, model)
        ds = ds.repeat()
        ds = ds.batch(batch_size, drop_remainder=self.use_tpu)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def handle_custom_dataprocess(self, ds, model):

        custom_data_process = getattr(model, "inputs_process", None)

        if custom_data_process is not None and callable(custom_data_process):
            ds = ds.map(custom_data_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds


    def handle_multiple_optimizers(self, model, optimizers):

        print("Processing multiple optimizers")

        multi_optimizer_layers_fn = getattr(model, "multi_optimizers_layers", None)

        if multi_optimizer_layers_fn is None or not callable(multi_optimizer_layers_fn):
            print("Warning, multi_optimizers_layers is not implemented, use optimizer at index = 0")
            return optimizers[0]

        layers_for_multi_optimizers = multi_optimizer_layers_fn()

        if layers_for_multi_optimizers is None:
            print("Warning, multi_optimizers_layers is not implemented, use optimizer at index = 0")
            return optimizers[0]

        num_optimizers = len(optimizers)
        num_layers = len(layers_for_multi_optimizers)

        if num_optimizers != num_layers:
            raise ValueError(f"Num of layers of multiple optimizers must equal to the number of optimizers, found {num_layers} vs {num_optimizers}")

        optimizer_layer_pair_list = []

        for i in range(num_optimizers):
            optimizer_layer_pair_list += [(optimizers[i], layers_for_multi_optimizers[i])]

        return MultiOptimizer(optimizer_layer_pair_list)