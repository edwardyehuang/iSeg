from keras import ops
from keras.optimizers import SGD

class SGD_EXT (SGD):

    def update_step(self, gradient, variable, learning_rate):
        learning_rate = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)

        if hasattr(variable, "lr_multiplier"):
            learning_rate = learning_rate * variable.lr_multiplier
            # print(f"lr_multiplier: {variable.lr_multiplier}")

        m = None
        if self.momentum != 0:
            m = self.momentums[self._get_variable_index(variable)]

        if m is not None:
            momentum = ops.cast(self.momentum, variable.dtype)
            self.assign(
                m,
                ops.subtract(
                    ops.multiply(m, momentum),
                    ops.multiply(gradient, learning_rate),
                ),
            )
            if self.nesterov:
                self.assign_add(
                    variable,
                    ops.subtract(
                        ops.multiply(m, momentum),
                        ops.multiply(gradient, learning_rate),
                    ),
                )
            else:
                self.assign_add(variable, m)
        else:
            self.assign_sub(variable, ops.multiply(gradient, learning_rate))