import tensorflow as tf
import keras

try:
    _LION = keras.optimizers.Lion
except AttributeError:
    try:
        _LION = tf.keras.optimizers.experimental.Lion
    except AttributeError:
        raise ImportError("Lion optimizer requires TensorFlow >= 2.11")

from iseg.utils.op_utils import replace_nan


class Lion_EXT(_LION):

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)

        if hasattr(variable, "lr_multiplier"):
            lr = lr * variable.lr_multiplier

        beta_1 = tf.cast(self.beta_1, variable.dtype)
        beta_2 = tf.cast(self.beta_2, variable.dtype)

        var_key = self._var_key(variable)
        m = self.momentums[self._index_dict[var_key]]

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients (use m as a buffer)
            m.assign(m * beta_1)
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1.0 - beta_1), gradient.indices
                )
            )
            variable.assign_sub(lr * tf.math.sign(m))

            m.assign(m * beta_2 / beta_1)
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1.0 - beta_2 / beta_1),
                    gradient.indices,
                )
            )
        else:
            # Dense gradients
            variable.assign_sub(
                lr * tf.math.sign(m * beta_1 + gradient * (1.0 - beta_1))
            )
            m.assign(m * beta_2 + gradient * (1.0 - beta_2))

    def _clip_gradients(self, grads):
        clipped_grads = []
        for g in grads:
            g = replace_nan(g, tf.cast(0.0, g.dtype))
            clipped_grads.append(g)
        grads = super()._clip_gradients(clipped_grads)
        return grads
