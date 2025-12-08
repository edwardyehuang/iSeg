import tensorflow as tf
import keras

from iseg.utils.keras_ops import get_all_layers_v2, set_bn_momentum
from iseg.modelhelper import ModelHelper


def get_no_weight_decay_layers_names_from_model (model):

    layers = get_all_layers_v2(model)

    excluded_name_list = [
        "bias", 
        "relative_position_bias_table", 
        "pos", 
        "patch_embed",
        "class_token",
        "logits",
    ]

    for layer in layers:
        
        if (isinstance(layer, keras.layers.LayerNormalization) or
            isinstance(layer, keras.layers.BatchNormalization)):

            excluded_name_list.append(layer.name)
        else:
            layer_type_name = layer.__class__.__name__.lower()

            if "norm" in layer_type_name:
                excluded_name_list.append(layer.name)

            if "logits" in layer_type_name:
                excluded_name_list.append(layer.name)
        
    return excluded_name_list


def exclude_no_weight_decay_layers_in_optimizer (
    optimizer, 
    model,
    excluded_name_list=None, 
    print_excluded_list=True
):

    if excluded_name_list is None:
        excluded_name_list = get_no_weight_decay_layers_names_from_model(
            model=model
        )

    if isinstance(optimizer, list):
        for opt in optimizer:
            exclude_no_weight_decay_layers_in_optimizer(
                opt, model, excluded_name_list, print_excluded_list
            )
    else:
        
        exclude_from_weight_decay_func = getattr(optimizer, "exclude_from_weight_decay", None)

        if exclude_from_weight_decay_func is None or not callable(exclude_from_weight_decay_func):
            return

        if print_excluded_list:
            print("Excluded vars for weight decay")

            for excluded_name in excluded_name_list:
                print(excluded_name)

        exclude_from_weight_decay_func(
            var_names=excluded_name_list
        )


def set_weights_lr_multiplier (var_list, lr_multiplier=1.0):

    if isinstance(var_list, tuple):
        var_list = list(var_list)

    if not isinstance(var_list, list):
        var_list = [var_list]

    for v in var_list:
        v.lr_multiplier = lr_multiplier

        if hasattr(v, "values"):
            for sub_v in v.values:
                sub_v.lr_multiplier = lr_multiplier