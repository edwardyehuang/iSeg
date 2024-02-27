from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from iseg.utils.slash_utils import replace_slash


if LooseVersion(tf.version.VERSION) < LooseVersion("2.13.0"):
    from keras.saving.legacy.hdf5_format import (
        load_attributes_from_hdf5_group,
        preprocess_weights_for_loading,
        _legacy_weights,
    )
elif LooseVersion(tf.version.VERSION) < LooseVersion("2.16.0"):
    from keras.src.saving.legacy.hdf5_format import (
        load_attributes_from_hdf5_group, 
        preprocess_weights_for_loading, 
        _legacy_weights
    )
else:
    from keras.src.legacy.saving.legacy_h5_format import (
        load_attributes_from_hdf5_group, 
        _legacy_weights
    )


from keras import backend
from absl import logging

def load_weights_from_hdf5_group_by_name(
        f, layers, skip_mismatch=False):
    """Implements name-based weight loading.

    (instead of topological weight loading).

    Layers that have no matching name are skipped.

    Args:
            f: A pointer to a HDF5 group.
            layers: a list of target layers.
            skip_mismatch: Boolean, whether to skip loading of layers
                    where there is a mismatch in the number of weights,
                    or a mismatch in the shape of the weights.

    Raises:
            ValueError: in case of mismatch between provided layers
                    and weights file and skip_match=False.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version']
        if hasattr(original_keras_version, 'decode'):
            original_keras_version = original_keras_version.decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend']
        if hasattr(original_backend, 'decode'):
            original_backend = original_backend.decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]

        name = replace_slash(name)

        for layer in index.get(name, []):
            symbolic_weights = _legacy_weights(layer)

            if LooseVersion(tf.version.VERSION) < LooseVersion("2.16.0"):
                weight_values = preprocess_weights_for_loading(
                        layer, weight_values, original_keras_version, original_backend)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    logging.warning('Skipping loading of weights for '
                                                    'layer {}'.format(layer.name) + ' due to mismatch '
                                                    'in number of weights ({} vs {}).'.format(
                                                            len(symbolic_weights), len(weight_values)))
                    continue
                raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                                                 '") expects ' + str(len(symbolic_weights)) +
                                                 ' weight(s), but the saved weights' + ' have ' +
                                                 str(len(weight_values)) + ' element(s).')
            # Set values.
            for i in range(len(weight_values)):
                if is_weights_mismatch(symbolic_weights[i], weight_values[i]):
                    if skip_mismatch:
                        logging.warning('Skipping loading of weights for '
                                                        'layer {}'.format(layer.name) + ' due to '
                                                        'mismatch in shape ({} vs {}).'.format(
                                                                symbolic_weights[i].shape,
                                                                weight_values[i].shape))
                        continue
                    raise ValueError('Layer #' + str(k) +' (named "' + layer.name +
                                                     '"), weight ' + str(symbolic_weights[i]) +
                                                     ' has shape {}'.format(
                                                             symbolic_weights[i].shape) +
                                                     ', but the saved weight has shape ' +
                                                     str(weight_values[i].shape) + '.')
                else:
                    weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
    batch_set_value(weight_value_tuples)


def is_weights_mismatch(symbolic_weight, weight_value):

    if LooseVersion(tf.version.VERSION) < LooseVersion("2.16.0"):
        return backend.int_shape(symbolic_weight) != weight_value.shape
    else:
        return symbolic_weight.shape != weight_value.shape
    


def dtype_numpy(x):
    """Returns the numpy dtype of a Keras tensor or variable.

    Args:
        x: Tensor or variable.

    Returns:
        numpy.dtype, dtype of `x`.
    """
    return tf.as_dtype(x.dtype).as_numpy_dtype



def get_graph():
    if tf.executing_eagerly():
        global _GRAPH
        if not getattr(_GRAPH, "graph", None):
            _GRAPH.graph = tf.__internal__.FuncGraph("keras_graph")
        return _GRAPH.graph
    else:
        return tf.compat.v1.get_default_graph()


def batch_set_value(tuples):
    """Sets the values of many tensor variables at once.

    Args:
        tuples: a list of tuples `(tensor, value)`.
            `value` should be a Numpy array.
    """
    if tf.executing_eagerly() or tf.inside_function():
        for x, value in tuples:
            x.assign(np.asarray(value, dtype=dtype_numpy(x)))
    else:
        with get_graph().as_default():
            if tuples:
                assign_ops = []
                feed_dict = {}
                for x, value in tuples:
                    value = np.asarray(value, dtype=dtype_numpy(x))
                    tf_dtype = tf.as_dtype(x.dtype.name.split("_")[0])
                    if hasattr(x, "_assign_placeholder"):
                        assign_placeholder = x._assign_placeholder
                        assign_op = x._assign_op
                    else:
                        # In order to support assigning weights to resizable
                        # variables in Keras, we make a placeholder with the
                        # correct number of dimensions but with None in each
                        # dimension. This way, we can assign weights of any size
                        # (as long as they have the correct dimensionality).
                        placeholder_shape = tf.TensorShape([None] * value.ndim)
                        assign_placeholder = tf.compat.v1.placeholder(
                            tf_dtype, shape=placeholder_shape
                        )
                        assign_op = x.assign(assign_placeholder)
                        x._assign_placeholder = assign_placeholder
                        x._assign_op = assign_op
                    assign_ops.append(assign_op)
                    feed_dict[assign_placeholder] = value
                tf.keras.backend.get_session().run(assign_ops, feed_dict=feed_dict)