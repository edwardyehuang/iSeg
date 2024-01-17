from distutils.version import LooseVersion
import tensorflow as tf

if LooseVersion(tf.version.VERSION) < LooseVersion("2.14.0"):
    _SGD = tf.keras.optimizers.experimental.SGD
else:
    _SGD = tf.keras.optimizers.SGD


class SGD_EXT (_SGD):

    def update_step(self, gradient, variable):
        
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)

        if hasattr(variable, "lr_multiplier"):
            lr = lr * variable.lr_multiplier
            # print(f"lr_multiplier: {variable.lr_multiplier}")

        m = None
        var_key = self._var_key(variable)
        momentum = tf.cast(self.momentum, variable.dtype)
        m = self.momentums[self._index_dict[var_key]]

        # TODO(b/204321487): Add nesterov acceleration.
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            add_value = tf.IndexedSlices(
                -gradient.values * lr, gradient.indices
            )
            if m is not None:
                m.assign(m * momentum)
                m.scatter_add(add_value)
                if self.nesterov:
                    variable.scatter_add(add_value)
                    variable.assign_add(m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.scatter_add(add_value)
        else:
            # Dense gradients
            if m is not None:
                m.assign(-gradient * lr + m * momentum)
                if self.nesterov:
                    variable.assign_add(-gradient * lr + m * momentum)
                else:
                    variable.assign_add(m)
            else:
                variable.assign_add(-gradient * lr)