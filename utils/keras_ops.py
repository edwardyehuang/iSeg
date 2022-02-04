# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import h5py

from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group_by_name, save_weights_to_hdf5_group

from tensorflow.python.framework import ops
from tensorflow.python.ops.parallel_for.control_flow_ops import pfor
from tensorflow.python.keras import backend as K

from iseg.layers import SyncBatchNormalization


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


def set_bn_momentum(model, momentum=0.99):

    layers = get_all_layers(model)

    for layer in layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = momentum
        elif isinstance(layer, SyncBatchNormalization):
            layer.momentum = momentum


def set_bn_epsilon(model, epsilon=1e-3):

    layers = get_all_layers(model)

    for layer in layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.epsilon = epsilon
        elif isinstance(layer, SyncBatchNormalization):
            layer.epsilon = epsilon


def save_model_to_h5(model: tf.keras.Model, path):
    with h5py.File(path, "w") as f:
        save_weights_to_hdf5_group(f, get_all_layers(model))


def load_h5_weight(model, path, skip_mismatch=False):

    with h5py.File(path, "r") as f:
        if "layer_names" not in f.attrs and "model_weights" in f:
            f = f["model_weights"]

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


class HookLayer(tf.keras.Model):
    def __init__(self, target):
        super(HookLayer, self).__init__()

        self.target = target

    def call(self, inputs, training=None):

        self.input_features = inputs

        x = inputs

        self.result = self.target(x, training=training)
        x = self.result

        return x
