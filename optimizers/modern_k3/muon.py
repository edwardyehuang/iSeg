from keras import ops
from keras.optimizers import Optimizer


def _matrix_transpose(x):
    """Transpose the last two dimensions of a tensor."""
    ndim = len(x.shape)
    perm = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
    return ops.transpose(x, axes=perm)


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Coefficients selected to maximize the slope at zero (Keller Jordan's values).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    transpose = False
    if ops.shape(X)[-2] > ops.shape(X)[-1]:
        X = _matrix_transpose(X)
        transpose = True
    X = X / (
        ops.sqrt(
            ops.sum(ops.square(X), axis=(-2, -1), keepdims=True)
        )
        + 1e-7
    )
    for _ in range(steps):
        A = X @ _matrix_transpose(X)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = _matrix_transpose(X)
    return X


class Muon_EXT(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz (Keras 3 backend-agnostic version).

    - Parameters with ndim >= 2 are updated via Muon (SGD-momentum + orthogonalization).
    - Parameters with ndim < 2 fall back to AdamW.

    Hyperparameters:
      lr: Muon learning rate (default 0.02).
      weight_decay: Muon weight decay (default 0.0).
      momentum: Muon momentum (default 0.95).
      nesterov: Use Nesterov momentum for Muon (default True).
      ns_steps: Newton-Schulz iteration steps (default 5).
      adamw_lr: Fallback AdamW learning rate (default 3e-4).
      adamw_beta_1: Fallback AdamW beta1 (default 0.9).
      adamw_beta_2: Fallback AdamW beta2 (default 0.95).
      adamw_weight_decay: Fallback AdamW weight decay (default 0.01).
      adamw_epsilon: Fallback AdamW epsilon (default 1e-8).
    """

    def __init__(
        self,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_lr=3e-4,
        adamw_beta_1=0.9,
        adamw_beta_2=0.95,
        adamw_weight_decay=0.01,
        adamw_epsilon=1e-8,
        clipnorm=None,
        clipvalue=None,
        name="Muon_EXT",
        **kwargs,
    ):
        super().__init__(
            learning_rate=lr,
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            **kwargs,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.adamw_lr = adamw_lr
        self.adamw_beta_1 = adamw_beta_1
        self.adamw_beta_2 = adamw_beta_2
        self.adamw_weight_decay = adamw_weight_decay
        self.adamw_epsilon = adamw_epsilon

    def build(self, variables):
        super().build(variables)
        if hasattr(self, "_muon_momentums"):
            return
        self._muon_momentums = []
        self._adamw_momentums = []
        self._adamw_velocities = []
        self._is_muon = []
        for var in variables:
            if len(var.shape) >= 2:
                self._muon_momentums.append(
                    self.add_variable_from_reference(
                        var, "muon_momentum"
                    )
                )
                self._is_muon.append(True)
            else:
                self._adamw_momentums.append(
                    self.add_variable_from_reference(
                        var, "adamw_momentum"
                    )
                )
                self._adamw_velocities.append(
                    self.add_variable_from_reference(
                        var, "adamw_velocity"
                    )
                )
                self._is_muon.append(False)

    def update_step(self, gradient, variable, learning_rate):
        idx = self._get_variable_index(variable)
        if self._is_muon[idx]:
            self._muon_update_step(gradient, variable, learning_rate, idx)
        else:
            self._adamw_update_step(gradient, variable, idx)

    def _muon_update_step(self, gradient, variable, learning_rate, idx):
        lr = ops.cast(learning_rate, variable.dtype)
        if hasattr(variable, "lr_multiplier"):
            lr = lr * variable.lr_multiplier

        m = self._muon_momentums[idx]
        momentum = ops.cast(self.momentum, variable.dtype)

        self.assign(
            m,
            m + (gradient - m) * (1 - momentum),
        )

        if self.nesterov:
            update = gradient + momentum * (m - gradient)
        else:
            update = m

        original_shape = ops.shape(update)
        rank = len(update.shape)
        if rank > 2:
            update = ops.reshape(
                update, [ops.shape(update)[0], -1]
            )

        update = zeropower_via_newtonschulz5(update, steps=self.ns_steps)

        rows = ops.cast(ops.shape(update)[-2], update.dtype)
        cols = ops.cast(ops.shape(update)[-1], update.dtype)
        scale = ops.sqrt(
            ops.maximum(ops.cast(1.0, update.dtype), rows / cols)
        )
        update = update * scale

        if rank > 2:
            update = ops.reshape(update, original_shape)

        if self.weight_decay != 0:
            self.assign_sub(
                variable,
                variable
                * ops.cast(self.lr * self.weight_decay, variable.dtype),
            )

        self.assign_sub(variable, update * lr)

    def _adamw_update_step(self, gradient, variable, idx):
        lr = ops.cast(self.adamw_lr, variable.dtype)
        if hasattr(variable, "lr_multiplier"):
            lr = lr * variable.lr_multiplier

        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(
            ops.cast(self.adamw_beta_1, variable.dtype), local_step
        )
        beta_2_power = ops.power(
            ops.cast(self.adamw_beta_2, variable.dtype), local_step
        )

        m = self._adamw_momentums[idx]
        v = self._adamw_velocities[idx]

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        self.assign(
            m,
            m + (gradient - m) * (1 - self.adamw_beta_1),
        )
        self.assign(
            v,
            v + (ops.square(gradient) - v) * (1 - self.adamw_beta_2),
        )

        if self.adamw_weight_decay != 0:
            self.assign_sub(
                variable,
                variable
                * ops.cast(
                    self.adamw_lr * self.adamw_weight_decay, variable.dtype
                ),
            )

        self.assign_sub(
            variable,
            (m * alpha) / (ops.sqrt(v) + self.adamw_epsilon),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                "ns_steps": self.ns_steps,
                "adamw_lr": self.adamw_lr,
                "adamw_beta_1": self.adamw_beta_1,
                "adamw_beta_2": self.adamw_beta_2,
                "adamw_weight_decay": self.adamw_weight_decay,
                "adamw_epsilon": self.adamw_epsilon,
            }
        )
        return config
