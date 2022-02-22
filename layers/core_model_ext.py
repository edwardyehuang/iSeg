# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

import iseg.static_strings as ss

from iseg.core_model import SegFoundation
from iseg.backbones.feature_extractor import get_backbone
from iseg.utils.common import resize_image


class SegManaged(SegFoundation):
    def __init__(
        self,
        backbone_name=ss.RESNET50,
        backbone_weights_path=None,
        output_stride=32,
        num_class=21,
        num_aux_loss=0,
        aux_loss_rate=0.4,
        aux_metric_names=None,
        custom_aux_loss_fns=[],
        use_ohem=False,
        ohem_thresh=0.7,
        label_as_inputs=False,
        label_as_backbone_inputs=False,
        label_as_head_inputs=False,
        use_custom_logits=False,
        logits_upsample_masks=None,
        resnet_multi_grids=[1, 2, 4],
        **kwargs,
    ):

        super().__init__(
            num_class=num_class,
            num_aux_loss=num_aux_loss,
            aux_loss_rate=aux_loss_rate,
            aux_metric_names=aux_metric_names,
            use_ohem=use_ohem,
            ohem_thresh=ohem_thresh,
            label_as_inputs=label_as_inputs,
            custom_aux_loss_fns=custom_aux_loss_fns,
            **kwargs,
        )

        self.backbone_name = backbone_name
        self.backbone_weights_path = backbone_weights_path
        self.output_stride = output_stride

        self.label_as_backbone_inputs = label_as_backbone_inputs
        self.label_as_head_inputs = label_as_head_inputs

        self.use_custom_logits = use_custom_logits
        self.logits_upsample_masks = logits_upsample_masks

        self.head = None

        label_shape = None

        if self.label_as_inputs and self.label_as_backbone_inputs:
            label_shape = (1, 513, 513)

        self.backbone = get_backbone(
            self.backbone_name,
            output_stride=self.output_stride,
            weights_path=self.backbone_weights_path,
            return_endpoints=True,
            image_shape=(1, 513, 513, 3),
            label_shape=label_shape,
            resnet_multi_grids=resnet_multi_grids,
        )

        if not self.use_custom_logits:
            self.logits_conv = tf.keras.layers.Conv2D(self.num_class, (1, 1), name=f"{self.name}/logits_conv")
            self.aux_logits_convs = []

            for i in range(self.num_aux_loss):
                prefix = "aux" if self.aux_metric_names is None else self.aux_metric_names[i]

                aux_logits_conv = tf.keras.layers.Conv2D(
                    self.num_class, (1, 1), name=f"{self.name}/{prefix}_logits_conv_{i}"
                )
                self.aux_logits_convs.append(aux_logits_conv)

        self.layers_for_multi_optimizers = None

    def call(self, inputs, training=None):

        x = inputs
        label = None

        if self.label_as_inputs:
            x, label = x

        inputs_size = tf.shape(x)[1:3]

        backbone_inputs = x

        if self.label_as_inputs and self.label_as_backbone_inputs:
            backbone_inputs = [backbone_inputs, label]

        endpoints = self.backbone(backbone_inputs, training=training)

        # Compute head results

        head_inputs = endpoints

        if self.label_as_inputs and self.label_as_head_inputs:
            head_inputs = [head_inputs, label]

        head_results = self.head(head_inputs, training=training)

        # Handle results

        if isinstance(head_results, tuple):
            head_results = list(head_results)

        if not isinstance(head_results, list):
            head_results = [head_results]

        assert len(head_results) == self.num_aux_loss + 1

        if not self.use_custom_logits:
            logits_list = [self.logits_conv(head_results[0])]

            for i in range(self.num_aux_loss):
                logits_list += [self.aux_logits_convs[i](head_results[i + 1])]
        else:
            logits_list = head_results

        if self.logits_upsample_masks is None:
            y = [tf.cast(resize_image(logits, inputs_size), tf.float32) for logits in logits_list]
        else:
            assert len(self.logits_upsample_masks) == len(logits_list)
            y = []

            for i in range(len(logits_list)):
                logits = logits_list[i]
                if self.logits_upsample_masks[i]:
                    logits = resize_image(logits, inputs_size)

                y += [tf.cast(logits, tf.float32)]

        if len(y) == 1:
            y = y[0]

        return y

    
    def multi_optimizers_layers(self):
        
        return self.layers_for_multi_optimizers

        #return super().multi_optimizers_layers()
