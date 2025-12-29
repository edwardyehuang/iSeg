from distutils.version import LooseVersion
import tensorflow as tf

if LooseVersion(tf.version.VERSION) < LooseVersion("2.14.0"):
    _ADAMW = tf.keras.optimizers.experimental.AdamW
else:
    _ADAMW = tf.keras.optimizers.AdamW

from iseg.utils.op_utils import replace_nan

class AdamW_EXT (_ADAMW):

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)

        if hasattr(variable, "lr_multiplier"):
            lr = lr * variable.lr_multiplier
            # print(f"lr_multiplier: {variable.lr_multiplier}")

        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - self.beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - self.beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))


    
    def _clip_gradients(self, grads):

        clipped_grads = []

        for g in grads:
            g = replace_nan(g, tf.cast(0.0, g.dtype))
            clipped_grads.append(g)

        
        grads = super()._clip_gradients(clipped_grads)

        return grads