from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from iseg.utils.slash_utils import replace_slash
from iseg.utils.version_utils import is_keras3

if LooseVersion(tf.version.VERSION) < LooseVersion("2.13.0"):
    from keras.saving.hdf5_format import (
        load_attributes_from_hdf5_group,
        preprocess_weights_for_loading,
        _legacy_weights,
    )
elif not is_keras3():
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
        f, layers, skip_mismatch=False
    ):
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

            if not is_keras3():
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

def load_subset_weights_from_hdf5_group(f):
    """Load layer weights of a model from hdf5.

    Args:
        f: A pointer to a HDF5 group.

    Returns:
        List of NumPy arrays of the weight values.

    Raises:
        ValueError: in case of mismatch between provided model
            and weights file.
    """
    weight_names = load_attributes_from_hdf5_group(f, "weight_names")
    return [np.asarray(f[weight_name]) for weight_name in weight_names]


def load_weights_from_hdf5_group_by_name_v2(f, model, skip_mismatch=False):
    """Implements name-based weight loading (instead of topological loading).

    Layers that have no matching name are skipped.

    Args:
        f: A pointer to a HDF5 group.
        model: Model instance.
        skip_mismatch: Boolean, whether to skip loading of layers
            where there is a mismatch in the number of weights,
            or a mismatch in the shape of the weights.

    Raises:
        ValueError: in case of mismatch between provided layers
            and weights file and skip_match=False.
    """
    if "keras_version" in f.attrs:
        original_keras_version = f.attrs["keras_version"]
        if hasattr(original_keras_version, "decode"):
            original_keras_version = original_keras_version.decode("utf8")
    else:
        original_keras_version = "1"
    if "backend" in f.attrs:
        original_backend = f.attrs["backend"]
        if hasattr(original_backend, "decode"):
            original_backend = original_backend.decode("utf8")
    else:
        original_backend = None

    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, "layer_names")

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in model.layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_values = load_subset_weights_from_hdf5_group(g)

        name = replace_slash(name)

        for layer in index.get(name, []):
            symbolic_weights = _legacy_weights(layer)

            if not is_keras3():
                weight_values = preprocess_weights_for_loading(
                    layer, weight_values, original_keras_version, original_backend
                )
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    logging.warning(
                        f"Skipping loading of weights for layer #{k} (named "
                        f"{layer.name}) due to mismatch in number of weights. "
                        f"Layer expects {len(symbolic_weights)} weight(s). "
                        f"Received {len(weight_values)} saved weight(s)"
                    )
                    continue
                raise ValueError(
                    f"Weight count mismatch for layer #{k} "
                    f"(named {layer.name}). "
                    f"Layer expects {len(symbolic_weights)} weight(s). "
                    f"Received {len(weight_values)} saved weight(s)"
                )
            # Set values.
            for i in range(len(weight_values)):
                expected_shape = _get_weight_shape(symbolic_weights[i])
                received_shape = weight_values[i].shape
                
                if expected_shape != received_shape:
                    if skip_mismatch:
                        logging.warning(
                            f"Skipping loading weights for layer #{k} (named "
                            f"{layer.name}) due to mismatch in shape for "
                            f"weight {symbolic_weights[i].name}. "
                            f"Weight expects shape {expected_shape}. "
                            "Received saved weight "
                            f"with shape {received_shape}"
                        )
                        continue
                    raise ValueError(
                        f"Shape mismatch in layer #{k} (named {layer.name}) "
                        f"for weight {symbolic_weights[i].name}. "
                        f"Weight expects shape {expected_shape}. "
                        "Received saved weight "
                        f"with shape {received_shape}"
                    )
                else:
                    weight_value_tuples.append(
                        (symbolic_weights[i], weight_values[i])
                    )

    if "top_level_model_weights" in f:

        if is_keras3():
            symbolic_weights = (model._trainable_variables + model._non_trainable_variables)
        else:
            symbolic_weights = (
                model._trainable_weights + model._non_trainable_weights
            )
        
        weight_values = load_subset_weights_from_hdf5_group(
            f["top_level_model_weights"]
        )

        if len(weight_values) != len(symbolic_weights):
            if skip_mismatch:
                logging.warning(
                    "Skipping loading top-level weights for model due to "
                    "mismatch in number of weights. "
                    f"Model expects {len(symbolic_weights)} "
                    "top-level weight(s). "
                    f"Received {len(weight_values)} saved top-level weight(s)"
                )
            else:
                raise ValueError(
                    "Weight count mismatch for top-level weights of model. "
                    f"Model expects {len(symbolic_weights)} "
                    "top-level weight(s). "
                    f"Received {len(weight_values)} saved top-level weight(s)"
                )
        else:
            for i in range(len(weight_values)):

                current_symbolic_weight = symbolic_weights[i]

                expected_shape = _get_weight_shape(current_symbolic_weight)
                received_shape = weight_values[i].shape

                if "pos_embed" in current_symbolic_weight.name:
                    expected_shape = received_shape

                if expected_shape != received_shape:
                    if skip_mismatch:
                        logging.warning(
                            "Skipping loading top-level weight for model due "
                            "to mismatch in shape for "
                            f"weight {symbolic_weights[i].name}. "
                            f"Weight expects shape {expected_shape}. "
                            "Received saved weight "
                            f"with shape {received_shape}"
                        )
                    else:
                        raise ValueError(
                            "Shape mismatch in model for top-level weight "
                            f"{symbolic_weights[i].name}. "
                            f"Weight expects shape {expected_shape}. "
                            "Received saved weight "
                            f"with shape {received_shape}"
                        )
                else:
                    weight_value_tuples.append(
                        (symbolic_weights[i], weight_values[i])
                    )

    batch_set_value(weight_value_tuples)

    if not is_keras3():
        # Perform any layer defined finalization of the layer state.
        for layer in model._flatten_layers():
            layer.finalize_state()


def is_weights_mismatch(symbolic_weight, weight_value):

    if not is_keras3():
        return backend.int_shape(symbolic_weight) != weight_value.shape
    else:
        return symbolic_weight.shape != weight_value.shape
    
def _get_weight_shape(weight):

    if not is_keras3():
        return backend.int_shape(weight)
    else:
        return weight.shape



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