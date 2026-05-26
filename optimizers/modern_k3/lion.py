from keras import ops
from keras.optimizers import Lion


class Lion_EXT(Lion):

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(learning_rate, variable.dtype)

        if hasattr(variable, "lr_multiplier"):
            lr = lr * variable.lr_multiplier
            print(f"lr_multiplier: {variable.name} {variable.lr_multiplier}")

        gradient = ops.cast(gradient, variable.dtype)
        beta_1 = ops.cast(self.beta_1, variable.dtype)
        beta_2 = ops.cast(self.beta_2, variable.dtype)

        m = self._momentums[self._get_variable_index(variable)]

        self.assign_sub(
            variable,
            ops.multiply(
                lr,
                ops.sign(
                    ops.add(
                        ops.multiply(m, beta_1),
                        ops.multiply(gradient, (1.0 - beta_1)),
                    )
                ),
            ),
        )
        self.assign(
            m,
            ops.add(
                ops.multiply(m, beta_2),
                ops.multiply(gradient, (1.0 - beta_2)),
            ),
        )
