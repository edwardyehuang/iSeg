# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

from distutils.version import LooseVersion
from functools import partial

import tensorflow as tf
import h5py

from tensorflow.python.keras.saving.hdf5_format import save_weights_to_hdf5_group
from iseg.utils.hdf5_utils import load_weights_from_hdf5_group_by_name, load_weights_from_hdf5_group_by_name_v2

from tensorflow.python.framework import ops
from tensorflow.python.ops.parallel_for.control_flow_ops import pfor
from tensorflow.python.keras import backend as K
from iseg.layers import SyncBatchNormalization
from iseg.utils.keras3_utils import Keras3_Model_Wrapper


def get_all_layers_v2(model, recursive=True, include_self=True):

    return list(getattr(model, "_flatten_layers")(recursive=recursive, include_self=include_self))


def get_all_layers(model):

    layers = []

    if hasattr(model, "layers"):
        for layer in model.layers:
            layers.extend(get_all_layers(layer))

    layers.append(model)

    return layers


def set_weight_decay(
    model, 
    weight_decay=0.0001, 
    decay_norm_vars=False,
    ):

    layers = get_all_layers(model)

    for layer in layers:
        if hasattr(layer, "kernel_regularizer"):
            layer.kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        
            if hasattr(layer, "kernel"):
                layer.kernel.regularizer = tf.keras.regularizers.l2(weight_decay)

        if hasattr(layer, "depthwise_regularizer"):
            layer.depthwise_regularizer = tf.keras.regularizers.l2(weight_decay)

        if decay_norm_vars:
            if hasattr(layer, "beta_regularizer"):
                layer.beta_regularizer = tf.keras.regularizers.l2(weight_decay)
            if hasattr(layer, "gamma_regularizer"):
                layer.gamma_regularizer = tf.keras.regularizers.l2(weight_decay)


def set_kernel_initializer(model, initializer):
    layers = get_all_layers(model)

    for layer in layers:
        if hasattr(layer, "kernel_initializer"):
            layer.kernel_initializer = initializer


def check_is_class_instance(obj, cls):

    cls = cls.func if isinstance(cls, partial) else cls

    return isinstance(obj, cls)


def set_bn_momentum(model, momentum=0.99):

    layers = get_all_layers(model)

    for layer in layers:
        if check_is_class_instance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = momentum
        elif check_is_class_instance(layer, SyncBatchNormalization):
            layer.momentum = momentum


def set_bn_epsilon(model, epsilon=1e-3):

    layers = get_all_layers(model)

    for layer in layers:
        if check_is_class_instance(layer, tf.keras.layers.BatchNormalization):
            layer.epsilon = epsilon
        elif check_is_class_instance(layer, SyncBatchNormalization):
            layer.epsilon = epsilon


def save_model_to_h5(model: tf.keras.Model, path):
    with h5py.File(path, "w") as f:
        save_weights_to_hdf5_group(f, get_all_layers(model))


def load_h5_weight(model, path, skip_mismatch=False, use_v2_behavior=False):

    with h5py.File(path, "r") as f:
        if "layer_names" not in f.attrs and "model_weights" in f:
            f = f["model_weights"]

        if use_v2_behavior:
            load_weights_from_hdf5_group_by_name_v2(f, model, skip_mismatch=skip_mismatch)
        else:
            layers = get_all_layers(model)
            load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=skip_mismatch)


def get_training_value(training=None):

    if training is None:
        training = K.learning_phase()

    if isinstance(training, int):
        training = bool(training)

    return training


def repeat_fn(fn, elems, global_elems, name=None):

    if name is None:
        name = "repeat_fn"

    with ops.name_scope(name):

        def loop_fn(i):
            gathered_elems = tf.nest.map_structure(lambda x: tf.gather(x, i), elems)
            return fn(gathered_elems + global_elems)

        batch_size = None
        first_elem = ops.convert_to_tensor(tf.nest.flatten(elems)[0])

        _list = first_elem.shape.as_list()

        return pfor(loop_fn, _list[0])


@tf.function(experimental_relax_shapes=True)
def multiply_with_sum(a, b, axis=0, name=None):
    with tf.name_scope(name if name is not None else "multiply_with_sum") as scope:

        b = tf.cast(b, a.dtype)

        a_shape = tf.shape(a)

        a_shape_list = a.shape.as_list()
        b_shape_list = b.shape.as_list()

        dims = len(a_shape_list)

        if axis < 0:
            axis = dims + axis

        final_shape = []

        for i in range(len(a_shape_list)):
            if i == axis:
                continue

            a_i = a_shape_list[i]
            b_i = b_shape_list[i]

            if a_i is None:
                final_shape.append(a_shape[i])
            else:
                final_shape.append(a_i if b_i == 1 else b_i)

        transpose_order = list(range(dims))
        transpose_order.remove(axis)
        transpose_order.insert(0, axis)

        a = tf.transpose(a, transpose_order)
        b = tf.transpose(b, transpose_order)

        result = tf.zeros(final_shape, dtype=a.dtype)

        for i in range(tf.shape(a)[0]):
            result += a[i] * b[i]

        return result


def capture_func(model, func_name):
    captured_func = getattr(model, func_name, None)

    if captured_func is not None and callable(captured_func):
        return captured_func

    return None


def replace_nan(x, value=0.0):
    return tf.where(tf.math.is_nan(x), tf.ones_like(x) * value, x)


def replace_inf(x):

    inf_to_zero = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
    max_value = tf.reduce_max(inf_to_zero)
    min_value = tf.reduce_min(inf_to_zero)

    return tf.clip_by_value(x, min_value, max_value)


def replace_nan_or_inf(x, nan_value=0.0):
    
    with tf.name_scope("replace_nan_or_inf"):
        return replace_inf(replace_nan(x, nan_value))


class HookLayer(Keras3_Model_Wrapper):
    def __init__(self, target):
        super(HookLayer, self).__init__()

        self.target = target

    def call(self, inputs, training=None):

        self.input_features = inputs

        x = inputs

        self.result = self.target(x, training=training)
        x = self.result

        return x
