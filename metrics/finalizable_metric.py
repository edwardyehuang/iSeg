# ================================================================
# MIT License
# Copyright (c) 2026 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import keras


class FinalizableMetric(keras.metrics.Metric):
    """Metric with an explicit end-of-evaluation notification.

    Subclasses can use `finalize_requested` to defer an expensive dataset-level
    result until the evaluation driver has processed the last batch. Keras
    resets the flag together with the other metric variables.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._finalize_requested = self.add_weight(
            name="finalize_requested",
            shape=[],
            initializer="zeros",
            dtype=tf.bool,
            synchronization=tf.VariableSynchronization.ON_WRITE,
            aggregation=tf.VariableAggregation.NONE,
        )

    @property
    def finalize_requested(self):
        return tf.convert_to_tensor(self._finalize_requested)

    def request_finalize(self):
        """Allows the next result call to perform the final computation."""
        return self._finalize_requested.assign(True)

    def reset_states(self):
        # Keras 2.15 assigns scalar zero to every variable, which fails for
        # array-valued metric buffers. Reset all state, including the flag,
        # with each variable's declared shape instead.
        for variable in self.variables:
            variable.assign(tf.zeros(variable.shape, variable.dtype))
