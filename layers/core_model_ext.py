# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

import iseg.static_strings as ss

from iseg.core_model import SegFoundation
from iseg.backbones.feature_extractor import get_backbone
from iseg.utils.common import resize_image
from iseg.utils.keras_ops import capture_func
from iseg.utils.version_utils import is_keras3
from iseg.utils.keras3_utils import _N
from iseg.utils.value_utils import values_to_list

from iseg.data_process.input_norm_types import InputNormTypes

import iseg.utils.common


class SegManaged(SegFoundation):
    def __init__(
        self,
        backbone_name=ss.RESNET50,
        backbone_weights_path=None,
        backbone_custom_fn=None,
        output_stride=32,
        num_class=21,
        input_norm_type=InputNormTypes.ZERO_MEAN,
        build_input_size=(512, 512),
        custom_main_loss_fn=None,
        num_aux_loss=0,
        aux_loss_rate=0.4,
        aux_metric_names=None,
        custom_aux_loss_fns=[],
        use_ohem=False,
        ohem_thresh=0.7,
        use_focal_loss=False,
        focal_loss_gamma=2.0,
        focal_loss_alpha=1.0,
        class_weights=None,
        label_as_inputs=False,
        label_as_backbone_inputs=False,
        label_as_head_inputs=False,
        image_as_head_inputs=False,
        use_custom_logits=False,
        logits_conv_postfix=None,
        logits_upsample_masks=None,
        resnet_multi_grids=[1, 2, 4],
        efficientnet_use_top=True,
        dict_inputs_image_key="image",
        backbone_outputs_dict_key="endpoints",
        head_results_direct_output=False,
        **kwargs,
    ):

        super().__init__(
            num_class=num_class,
            input_norm_type=input_norm_type,
            custom_main_loss_fn=custom_main_loss_fn,
            num_aux_loss=num_aux_loss,
            aux_loss_rate=aux_loss_rate,
            aux_metric_names=aux_metric_names,
            use_ohem=use_ohem,
            ohem_thresh=ohem_thresh,
            use_focal_loss=use_focal_loss,
            focal_loss_gamma=focal_loss_gamma,
            focal_loss_alpha=focal_loss_alpha,
            class_weights=class_weights,
            label_as_inputs=label_as_inputs,
            custom_aux_loss_fns=custom_aux_loss_fns,
            **kwargs,
        )

        self.backbone_name = backbone_name
        self.backbone_weights_path = backbone_weights_path
        self.output_stride = output_stride

        self.label_as_backbone_inputs = label_as_backbone_inputs
        self.label_as_head_inputs = label_as_head_inputs

        self.image_as_head_inputs = image_as_head_inputs

        self.use_custom_logits = use_custom_logits
        self.logits_upsample_masks = logits_upsample_masks

        self.head = None

        label_shape = None

        align_corners = iseg.utils.common.DEFAULT_ALIGN_CORNERS

        build_input_size = list(build_input_size)
        build_input_height = build_input_size[0]
        build_input_width = build_input_size[1]

        if align_corners:
            build_input_height += 1
            build_input_width += 1

        if self.label_as_inputs and self.label_as_backbone_inputs:
            label_shape = (1, build_input_height, build_input_width)

        image_shape = (1, build_input_height, build_input_width, 3)

        self.backbone = get_backbone(
            self.backbone_name,
            custom_backbone_fn=backbone_custom_fn,
            output_stride=self.output_stride,
            weights_path=self.backbone_weights_path,
            return_endpoints=True,
            image_shape=image_shape,
            label_shape=label_shape,
            resnet_multi_grids=resnet_multi_grids,
            efficientnet_use_top=efficientnet_use_top,
        )

        if not self.use_custom_logits:

            logits_conv_name = "logits_conv"

            if logits_conv_postfix is not None:
                logits_conv_name = f"{logits_conv_name}_{logits_conv_postfix}"

            self.logits_conv = tf.keras.layers.Conv2D(self.num_class, (1, 1), name=_N(f"{self.name}/{logits_conv_name}"))
            self.aux_logits_convs = self.build_aux_logits_conv(self.num_aux_loss, self.aux_metric_names)

        self.layers_for_multi_optimizers = None

        self.dict_inputs_image_key = dict_inputs_image_key
        self.backbone_outputs_dict_key = backbone_outputs_dict_key

        self.head_results_direct_output = head_results_direct_output


    def build_aux_logits_conv (self, num_aux_loss, aux_metric_names=None):

        aux_logits_convs = []

        for i in range(num_aux_loss):

            prefix = "aux" if aux_metric_names is None else aux_metric_names[i]

            aux_logits_conv = tf.keras.layers.Conv2D(
                self.num_class, (1, 1), name=_N(f"{self.name}/{prefix}_logits_conv_{i}")
            )
            aux_logits_convs.append(aux_logits_conv)


        return aux_logits_convs
    

    def compute_backbone_results (self, backbone_inputs, training=None):

        return self.backbone(backbone_inputs, training=training)
        

    def compute_head_results (self, head_inputs, training=None):

        #  replace None in head_inputs with 0

        for i in range(len(head_inputs)):
            if head_inputs[i] is None:
                head_inputs[i] = tf.zeros_like((), dtype=tf.float32)

        head_results = self.head(head_inputs, training=training)

        if isinstance(head_results, tuple):
            head_results = list(head_results)

        if not isinstance(head_results, list):
            head_results = [head_results]

        # assert len(head_results) == self.num_aux_loss + 1

        return head_results


    def compute_logits_results(self, logits_inputs):
        
        if not self.use_custom_logits:
            logits_list = [self.logits_conv(logits_inputs[0])]

            # Num of logits may < num of aux loss
            for i in range(len(self.aux_logits_convs)):
                logits_list += [self.aux_logits_convs[i](logits_inputs[i + 1])]
        else:
            logits_list = logits_inputs

        return logits_list


    def compute_logits_upsample(self, logits_list, inputs_size):

        if self.logits_upsample_masks is None:
            y = [self.upsample_single_logits(logits, inputs_size) for logits in logits_list]
        else:
            assert len(self.logits_upsample_masks) == len(logits_list)
            y = []

            for i in range(len(logits_list)):
                logits = logits_list[i]
                if self.logits_upsample_masks[i]:                 
                    logits = self.upsample_single_logits(logits, inputs_size)

                y += [logits]

        return y
    

    def upsample_single_logits (self, logits, target_size):

        resize_method = "bilinear"
                    
        if logits.dtype is tf.int32:
            resize_method = "nearest"

        logits = resize_image(logits, target_size, method=resize_method)

        return logits


    def compute_final_results (self, logits_list):

        y = [tf.cast(logits, tf.float32) for logits in logits_list]

        '''
        if len(y) == 1:
            y = y[0]
        '''

        if isinstance(y, list) and is_keras3():
            _y = y
            y = dict()

            for i in range(len(_y)):
                output_name = self._index_to_output_key(i)
                y[output_name] = tf.identity(_y[i], name=output_name)


        return y

    def call(self, inputs, training=None):

        return self._call_internal(inputs, training=training)
    

    @tf.autograph.experimental.do_not_convert
    def _call_internal(self, inputs, training=None):

        x = inputs
        label = None

        is_dict_inputs = isinstance(x, dict)

        if self.label_as_inputs:
            if not is_dict_inputs:
                x, label = x
            else:
                inputs_dict = x.copy()
                x = inputs_dict[self.dict_inputs_image_key]
                del inputs_dict[self.dict_inputs_image_key]
                label = inputs_dict

        image_tensor = tf.identity(x, name="image_tensor")
        inputs_size = tf.shape(image_tensor)[1:3]

        if is_dict_inputs:
            image_tensor = {self.dict_inputs_image_key: image_tensor}

        backbone_inputs = self.build_sub_model_inputs(
            image_tensor, 
            label, 
            with_label=self.label_as_inputs and self.label_as_backbone_inputs
        )

        backbone_inputs = self.extract_if_single_element(backbone_inputs)
        endpoints = self.compute_backbone_results(backbone_inputs, training=training)

        if is_dict_inputs:
            endpoints = {self.backbone_outputs_dict_key: endpoints}
        else:
            endpoints = [endpoints] # for backward compatibility

        # Compute head results

        head_inputs = self.build_sub_model_inputs(
            endpoints, 
            label, 
            with_label=self.label_as_inputs and self.label_as_head_inputs
        )

        if self.image_as_head_inputs:
            # head_inputs += [image_tensor]
            head_inputs = self.build_sub_model_inputs(
                head_inputs, 
                image_tensor, 
                with_label=True
            )

        head_inputs = self.extract_if_single_element(head_inputs)
        head_results = self.compute_head_results(head_inputs, training=training)

        if self.head_results_direct_output:
            return head_results

        # Handle logits

        logits_list = self.compute_logits_results(head_results)
        logits_list = self.compute_logits_upsample(logits_list, inputs_size=inputs_size)

        logits_list = self.compute_final_results(logits_list)

        return logits_list
    

    def build_sub_model_inputs(
        self, 
        inputs, 
        label=None,
        with_label=False
    ):
        
        if not with_label:
            return inputs

        is_dict_inputs = isinstance(inputs, dict)
        is_dict_label = isinstance(label, dict)

        assert is_dict_inputs == is_dict_label, "Inputs and label must be both dict or not dict"

        if not is_dict_inputs:
            inputs = values_to_list(inputs)
            label = values_to_list(label)

            return inputs + label
        
        inputs_dict = inputs.copy()
        inputs_dict.update(label)

        return inputs_dict
    

    def extract_if_single_element(self, possible_collection):

        if isinstance(possible_collection, (list, tuple, dict)) and len(possible_collection) == 1:
            if isinstance(possible_collection, dict):
                return list(possible_collection.values())[0]
            
            return possible_collection[0]
        
        return possible_collection
        
    
    def multi_optimizers_layers(self):
        
        return self.layers_for_multi_optimizers

        #return super().multi_optimizers_layers()


    def on_epoch_end(self, epoch, logs={}):

        head_on_epoch_end_func = capture_func(self.head, "on_epoch_end")

        if head_on_epoch_end_func is not None:
            head_on_epoch_end_func(epoch, logs)

        backbone_on_epoch_end_func = capture_func(self.backbone, "on_epoch_end")

        if backbone_on_epoch_end_func is not None:
            backbone_on_epoch_end_func(epoch, logs)