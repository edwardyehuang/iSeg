# This script implement name-based h5 loading without batch loading

import keras
import tensorflow as tf
import numpy as np

from keras import backend
from absl import logging

import h5py

from distutils.version import LooseVersion


from iseg.utils.slash_utils import replace_slash
from iseg.utils.version_utils import is_keras3

from iseg.utils.hdf5_utils import batch_set_value, _get_weight_shape, load_subset_weights_from_hdf5_group

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


def load_weights_from_hdf5_group_by_name_v3(f, model, skip_mismatch=False):
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

        layer_list = index.get(name, [])

        for layer in layer_list:
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


def load_subset_weights_kv_dict_from_hdf5_group(f):
    """Load layer weights kv dicts of a model from hdf5.

    Args:
        f: A pointer to a HDF5 group.

    Returns:
        Dictionary of NumPy arrays of the weight key-value pairs.

    Raises:
        ValueError: in case of mismatch between provided model
            and weights file.
    """
    weight_names = load_attributes_from_hdf5_group(f, "weight_names")

    weight_kv_dict = {}

    for weight_name in weight_names:
        g = f[weight_name]
        weight_kv_dict[weight_name] = np.asarray(g)

    return weight_kv_dict