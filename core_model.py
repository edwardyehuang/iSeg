# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.engine import data_adapter

from iseg.core_inference import *
from iseg.utils.common import resize_image, get_scaled_size, get_tensor_shape, smart_where

from iseg.metrics.utils import SegMetricBuilder
from iseg.losses.catecrossentropy_ignore_label import catecrossentropy_ignore_label_loss
from iseg.losses.ohem import get_ohem_fn
from iseg.utils.keras3_utils import Keras3_Model_Wrapper


class SegBase(Keras3_Model_Wrapper):

    

    def __init__(self, num_class=21, **kwargs):

        super(SegBase, self).__init__(**kwargs)

        self.num_class = num_class
        self.inference_sliding_window_size = None

    def predict_step(self, data):

        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)

        return self.inference(x, training=False)

    def test_step(self, data):

        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self.inference(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def inference(self, inputs, training=False):

        results = inference_fn(
            inputs,
            model=self,
            num_class=self.num_class,
            training=training,
            sliding_window_crop_size=self.inference_sliding_window_size,
        )

        return results

    @tf.function(autograph=False)
    def inference_with_scale(
        self, 
        inputs, 
        training=False, 
        scale_rate=1.0, 
        flip=False, 
        resize_method="bilinear"
    ):

        inputs_size = get_tensor_shape(inputs, return_list=True)[1:3]
        
        if flip:
            inputs = tf.image.flip_left_right(inputs)

        sizes = get_scaled_size(inputs, scale_rate, pad_mode=1)

        inputs = tf.cast(inputs, tf.float32)
        inputs = resize_image(inputs, sizes, method=resize_method, name="inference_resize")

        sliding_window_size = self.inference_sliding_window_size

        if sliding_window_size is not None:
            sliding_window_h = smart_where(sliding_window_size[0] < sizes[0], sliding_window_size[0], sizes[0])
            sliding_window_w = smart_where(sliding_window_size[1] < sizes[1], sliding_window_size[1], sizes[1])

            sliding_window_size = (sliding_window_h, sliding_window_w)

        logits = inference_fn(
            inputs,
            model=self,
            num_class=self.num_class,
            training=training,
            sliding_window_crop_size=sliding_window_size,
        )  # Under solving  #47261

        logits = convert_to_list_if_single(logits)

        logits = multi_results_handler(
            logits, lambda x: resize_image(x, inputs_size, method=resize_method, name="inference_resize_back")
        )

        logits = multi_results_handler(
            logits, lambda x: tf.image.flip_left_right(x) if flip else x
        )

        logits = free_from_list_if_single(logits)

        return logits

    def inference_with_multi_scales(
        self, 
        inputs, 
        training=False, 
        scale_rates=[1.0], 
        flip=False,
        resize_method="bilinear",
    ):
        num_rates = len(scale_rates)

        divide_factor = num_rates

        if flip:
            divide_factor *= 2

        logits_sum_list = None

        @tf.function(autograph=False)
        def loop_body(image, scale_rate=1.0, inner_flip=False):

            logits_list = self.inference_with_scale(
                image, 
                training=training, 
                scale_rate=scale_rate, 
                flip=inner_flip,
                resize_method=resize_method,
            )

            return convert_to_list_if_single(logits_list)

        logits_sum_list = loop_body(
            inputs, 
            tf.constant(scale_rates[0]), 
            inner_flip=False
        )

        for i in range(1, num_rates):
            logits_list = loop_body(
                inputs, 
                tf.constant(scale_rates[i]), 
                inner_flip=False
            )
            logits_sum_list = multi_results_add(logits_sum_list, logits_list)

        if flip:
            inputs = tf.image.flip_left_right(inputs)
            logits_sum_list = multi_results_handler(logits_sum_list, lambda x: tf.image.flip_left_right(x))

            for i in range(0, num_rates):
                logits_list = loop_body(inputs, tf.constant(scale_rates[i]), inner_flip=False)
                logits_sum_list = multi_results_add(logits_sum_list, logits_list)

            logits_sum_list = multi_results_handler(logits_sum_list, lambda x: tf.image.flip_left_right(x))

        result = [tf.math.divide(logits_sum, divide_factor) for logits_sum in logits_sum_list]

        result = free_from_list_if_single(result)

        return result


class SegFoundation(SegBase):
    def __init__(
        self,
        num_class=21,
        num_aux_loss=0,
        aux_loss_rate=0.4,
        aux_metric_names=None,
        aux_metric_iou_masks=None,
        aux_metric_pre_fns=[],
        use_ohem=False,
        ohem_thresh=0.7,
        label_as_inputs=False,
        custom_aux_loss_fns=[],
        use_focal_loss=False,
        focal_loss_gamma=2.0,
        focal_loss_alpha=1.0,
        class_weights=None,
        **kwargs,
    ):

        super().__init__(num_class=num_class, **kwargs)

        assert num_aux_loss >= 0, f"num_aux_loss must >= 0, found {num_aux_loss}"

        self.num_aux_loss = num_aux_loss

        if isinstance(aux_loss_rate, tuple):
            aux_loss_rate = list(aux_loss_rate)

        if not isinstance(aux_loss_rate, list):
            aux_loss_rate = [aux_loss_rate] * num_aux_loss

        assert (
            len(aux_loss_rate) == num_aux_loss
        ), f"aux_loss_rate must be scalar or has length = num_aux_loss, found {len(aux_loss_rate)}"

        if num_aux_loss == 0:
            aux_metric_names = None

        assert (aux_metric_names is None) or (
            len(aux_metric_names) == num_aux_loss
        ), f"aux_metric_names must be None or has equal length = num_aux_loss, found {len(aux_metric_names)}"


        self.aux_loss_rate = aux_loss_rate
        self.use_ohem = use_ohem
        self.ohem_thresh = ohem_thresh
        self.aux_metric_names = aux_metric_names
        self.aux_metric_iou_masks = aux_metric_iou_masks
        self.aux_metric_pre_fns = aux_metric_pre_fns
        self.label_as_inputs = label_as_inputs

        self.custom_aux_loss_fns = custom_aux_loss_fns

        self.use_focal_loss = use_focal_loss
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha

        self.model_class_weights = class_weights


    def inputs_process(self, image, label):

        if self.label_as_inputs:
            image = (image, label)

        if self.num_aux_loss > 0:
            label = tuple([label] * (self.num_aux_loss + 1))

        return image, label

    def __aux_index_to_output_key(self, index):

        key = index + 2
        key = f"output_{key}"

        return key
    

    def add_class_weights (
        self,
        class_weights=None,
        new_class_weights=None,
    ):

        if new_class_weights is not None:
            new_class_weights = np.array(new_class_weights)

            if class_weights is not None:
                class_weights *= new_class_weights
            else:
                class_weights = new_class_weights

        return class_weights



    def custom_losses(
        self, 
        num_class, 
        ignore_label,
        batch_size,
        class_weights=None,
        reduction=False, 
        **kwargs):

        ohem_func = get_ohem_fn(thresh=self.ohem_thresh) if self.use_ohem else None

        class_weights = self.add_class_weights(
            new_class_weights=class_weights
        )
        class_weights = self.add_class_weights(
            class_weights=class_weights, 
            new_class_weights=self.model_class_weights
        )

        common_kwargs = {
            "num_class": num_class,
            "ignore_label": ignore_label,
            "batch_size": batch_size,
            "reduction": reduction,
            "class_weights":class_weights,
        }

        loss = lambda post_func: catecrossentropy_ignore_label_loss(
            post_compute_fn=post_func, 
            use_focal_loss=self.use_focal_loss,
            focal_loss_gamma=self.focal_loss_gamma,
            focal_loss_alpha=self.focal_loss_alpha,
            **common_kwargs, 
            **kwargs,
        )

        loss_dict = {"output_1": loss(ohem_func)}

        if self.custom_aux_loss_fns is None or len(self.custom_aux_loss_fns) == 0:
            for i in range(self.num_aux_loss):
                loss_dict[self.__aux_index_to_output_key(i)] = loss(None)
        else:
            assert (
                len(self.custom_aux_loss_fns) == self.num_aux_loss
            ), "custom_aux_loss_fns must be None or empty, or has same length with num_aux_loss"

            for i in range(self.num_aux_loss):
                loss_dict[self.__aux_index_to_output_key(i)] = (
                    self.custom_aux_loss_fns[i](**common_kwargs, **kwargs)
                    if self.custom_aux_loss_fns[i] is not None
                    else loss(None)
                )

        return loss_dict

    def custom_losses_weights(self):

        weights_dict = {"output_1": 1.0}

        for i in range(self.num_aux_loss):
            weights_dict[self.__aux_index_to_output_key(i)] = self.aux_loss_rate[i]

        return weights_dict

    def custom_metrics(self, num_class, ignore_label):

        metrics = SegMetricBuilder(num_class, ignore_label)
        metrics.add()

        # Rest of the code is for aux metrics

        # IOU masks for aux metrics
        aux_metric_iou_masks = self.aux_metric_iou_masks

        if aux_metric_iou_masks is None or len(aux_metric_iou_masks) == 0:
            aux_metric_iou_masks = [False] * self.num_aux_loss

        assert len(aux_metric_iou_masks) == self.num_aux_loss

        # Pre_compute_fns for aux metrics
        aux_metric_pre_fns = self.aux_metric_pre_fns

        if aux_metric_pre_fns is None or len(aux_metric_pre_fns) == 0:
            aux_metric_pre_fns = [None] * self.num_aux_loss

        assert len(aux_metric_pre_fns) == self.num_aux_loss

        # Build aux metrics

        for i in range(self.num_aux_loss):

            prefix = "aux" if self.aux_metric_names is None else self.aux_metric_names[i]
            metrics.add(
                f"{prefix}_{i}", 
                use_iou=aux_metric_iou_masks[i],
                pre_compute_fn=aux_metric_pre_fns[i]
                )

        return metrics.metrics

    def multi_optimizers_layers(self):

        return None
