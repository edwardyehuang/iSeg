# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

import iseg.static_strings as ss

from iseg.core_model import SegFoundation
from iseg.backbones.feature_extractor import get_backbone
from iseg.utils.common import resize_image


class SegManaged (SegFoundation):

    def __init__(self, backbone_name = ss.RESNET50, 
                       backbone_weights_path = None,
                       output_stride = 32,
                       num_class = 21, 
                       num_aux_loss = 0, 
                       aux_loss_rate = 0.4,
                       aux_metric_names = None,
                       use_ohem = False, 
                       ohem_thresh = 0.7, 
                       label_as_inputs = False,
                       label_as_backbone_inputs = False,
                       label_as_head_inputs = False,
                       **kwargs):

        super().__init__(num_class = num_class, 
                         num_aux_loss = num_aux_loss, 
                         aux_loss_rate = aux_loss_rate, 
                         aux_metric_names = aux_metric_names,
                         use_ohem = use_ohem, 
                         ohem_thresh = ohem_thresh,
                         label_as_inputs = label_as_inputs,
                         **kwargs)

        self.backbone_name = backbone_name
        self.backbone_weights_path = backbone_weights_path
        self.output_stride = output_stride

        self.label_as_backbone_inputs = label_as_backbone_inputs
        self.label_as_head_inputs = label_as_head_inputs

        self.head = None

        label_shape = None

        if self.label_as_inputs and self.label_as_backbone_inputs:
            label_shape = (1, 513, 513)
        
        self.backbone = get_backbone(self.backbone_name,
                                     output_stride = self.output_stride,
                                     weights_path = self.backbone_weights_path,
                                     return_endpoints = True,
                                     image_shape = (1, 513, 513, 3),
                                     label_shape = label_shape)
        
        self.logits_conv = tf.keras.layers.Conv2D(self.num_class, (1, 1), name = f"{self.name}/logits_conv")
        self.aux_logits_convs = []

        for i in range(self.num_aux_loss):
            prefix = "aux" if self.aux_metric_names is None else self.aux_metric_names[i]

            aux_logits_conv = tf.keras.layers.Conv2D(self.num_class, (1, 1), name = f"{self.name}/{prefix}_logits_conv_{i}")
            self.aux_logits_convs.append(aux_logits_conv)


    def call (self, inputs, training = None):

        x = inputs
        label = None
        
        if self.label_as_inputs:
            x, label = x

        inputs_size = tf.shape(x)[1:3]

        backbone_inputs = x

        if self.label_as_inputs and self.label_as_backbone_inputs:
            backbone_inputs = [backbone_inputs, label]

        endpoints = self.backbone(backbone_inputs, training = training)
 
        # Compute head results

        head_inputs = endpoints

        if self.label_as_inputs and self.label_as_head_inputs:
            head_inputs = [head_inputs, label]

        head_results = self.head(head_inputs, training = training)

        # Handle results

        if isinstance(head_results, tuple):
            head_results = list(head_results)

        if not isinstance(head_results, list):
            head_results = [head_results]

        assert len(head_results) == self.num_aux_loss + 1

        logits_list = [self.logits_conv(head_results[0])]

        for i in range(self.num_aux_loss):
            logits_list += [self.aux_logits_convs[i](head_results[i + 1])]

        y = [tf.cast(resize_image(logits, inputs_size), tf.float32) for logits in logits_list]
        

        if len(y) == 1:
            y = y[0]

        return y