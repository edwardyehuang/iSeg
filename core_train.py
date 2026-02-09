# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import keras
import numpy as np
import math


from iseg.metrics.mean_iou import MeanIOU
from iseg.metrics.seg_metric_wrapper import SegMetricWrapper
from iseg.losses.catecrossentropy_ignore_label import catecrossentropy_ignore_label_loss
from iseg.callbacks.ckpt_saver import CheckpointSaver
from iseg.callbacks.time_callback import TimeCallback
from iseg.callbacks.model_callback import ModelCallback

from iseg.utils.model_utils import create_compiled_model


class CoreTrain(object):
    def __init__(
        self, 
        model_helper, 
        train_dataset, 
        val_dataset=None, 
        val_image_count=0, 
        use_tpu=False,
        use_tpu_pod=False,
        use_data_shared_policy_for_train=True,
        use_data_shared_policy_for_val=True,
    ):

        self.model_helper = model_helper
        self.training_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_image_count = val_image_count

        self.use_tpu = use_tpu
        self.use_tpu_pod = use_tpu_pod

        self.use_data_shared_policy_for_train = use_data_shared_policy_for_train
        self.use_data_shared_policy_for_val = use_data_shared_policy_for_val


    def create_trainable_model(
        self, 
        num_class, 
        ignore_label=255,
        class_weights=None, 
        batch_size=1, 
        epoch_steps=1000, 
        initial_epoch=0,
        jit_compile=None,
    ):

        model = self.model_helper.model
        optimizer = self.model_helper.optimizer

        return create_compiled_model(
            model=model,
            num_class=num_class, 
            ignore_label=ignore_label,
            class_weights=class_weights,
            batch_size=batch_size,
            epoch_steps=epoch_steps,
            initial_epoch=initial_epoch,
            jit_compile=jit_compile,
            optimizer=optimizer,
        )


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
        use_profiler=False,
        verbose=1,
        validation_freq=1,
        jit_compile=None,
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
                jit_compile=jit_compile,
            )

        if initial_epoch == -1:
            
            iteration_object = model.optimizer.iterations

            if hasattr(iteration_object, "value") and callable(iteration_object.value):
                current_iter = iteration_object.value()
            else:
                current_iter = iteration_object

            initial_epoch = current_iter // epoch_steps

        train_ds = self.prepare_train_dataset(model, batch_size, shuffle_rate)
        eval_ds = self.prepare_val_dataset(model, eval_batch_size)

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir, 
            histogram_freq=0, 
            write_images=False,
            profile_batch=0 if not use_profiler else (int(epoch_steps * 0.1), int(epoch_steps * 0.1) + 2),
        )

        checkpoint_saver = CheckpointSaver(self.model_helper)
        model_callback = ModelCallback(self.model_helper.model)

        val_steps = None if eval_ds is None else int(math.ceil(self.val_image_count / eval_batch_size))

        model_callbacks = [
            tensorboard_callback,
            checkpoint_saver, 
            model_callback, 
            TimeCallback(),
        ]

        # Note, we do not apply the shuffle in keras.model.fit as it has already shuffled in tf.data
        model.fit(
            train_ds,
            epochs=train_epoches,
            validation_data=eval_ds,
            shuffle=False,
            callbacks=model_callbacks,
            initial_epoch=initial_epoch,
            steps_per_epoch=epoch_steps,
            validation_steps=val_steps,
            verbose=verbose,
            validation_freq=validation_freq,
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

        ds = self.data_based_shard_policy(ds, self.use_data_shared_policy_for_train)

        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def prepare_val_dataset(self, model, batch_size=1):

        if self.val_dataset is None:
            return None

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        ds = self.handle_custom_dataprocess(self.val_dataset, model)
        ds = ds.repeat()
        ds = ds.batch(batch_size, drop_remainder=self.use_tpu)

        ds = self.data_based_shard_policy(ds, self.use_data_shared_policy_for_val)

        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds
    

    def data_based_shard_policy(self, ds, use_data_shared_policy=True):
        
        if self.use_tpu and self.use_tpu_pod and use_data_shared_policy:
            print("Use TPU pod! Set AutoShardPolicy to DATA")

            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            ds = ds.with_options(options)

        return ds


    def handle_custom_dataprocess(self, ds, model):

        custom_data_process = getattr(model, "inputs_process", None)

        if custom_data_process is not None and callable(custom_data_process):
            ds = ds.map(custom_data_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds