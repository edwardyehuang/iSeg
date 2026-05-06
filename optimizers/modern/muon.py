from distutils.version import LooseVersion
import tensorflow as tf

from iseg.utils.op_utils import replace_nan

if LooseVersion(tf.version.VERSION) < LooseVersion("2.14.0"):
    _OPTIMIZER = tf.keras.optimizers.Optimizer
else:
    _OPTIMIZER = tf.keras.optimizers.Optimizer


def _matrix_transpose(x):
    """Transpose the last two dimensions of a tensor."""
    ndim = len(x.shape)
    perm = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
    return tf.transpose(x, perm=perm)


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Coefficients selected to maximize the slope at zero (Keller Jordan's values).
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    transpose = False
    if X.shape[-2] > X.shape[-1]:
        X = _matrix_transpose(X)
        transpose = True
    X = X / (
        tf.sqrt(tf.reduce_sum(tf.square(X), axis=(-2, -1), keepdims=True)) + 1e-7
    )
    for _ in range(steps):
        A = X @ _matrix_transpose(X)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = _matrix_transpose(X)
    return X


class Muon_EXT(_OPTIMIZER):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    This implementation follows Keller Jordan's Muon optimizer.
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

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_muon_momentums_dict"):
            return

        self._muon_momentums_dict = {}
        self._adamw_momentums_dict = {}
        self._adamw_velocities_dict = {}
        self._use_muon_dict = {}

        for var in var_list:
            var_key = self._var_key(var)
            if len(var.shape) >= 2:
                self._muon_momentums_dict[var_key] = (
                    self.add_variable_from_reference(var, "muon_momentum")
                )
                self._use_muon_dict[var_key] = True
            else:
                self._adamw_momentums_dict[var_key] = (
                    self.add_variable_from_reference(var, "adamw_momentum")
                )
                self._adamw_velocities_dict[var_key] = (
                    self.add_variable_from_reference(var, "adamw_velocity")
                )
                self._use_muon_dict[var_key] = False

    def update_step(self, gradient, variable):
        var_key = self._var_key(variable)
        if self._use_muon_dict[var_key]:
            self._muon_update_step(gradient, variable, var_key)
        else:
            self._adamw_update_step(gradient, variable, var_key)

    def _muon_update_step(self, gradient, variable, var_key):
        lr = tf.cast(self.learning_rate, variable.dtype)
        if hasattr(variable, "lr_multiplier"):
            lr = lr * variable.lr_multiplier

        momentum = tf.cast(self.momentum, variable.dtype)
        m = self._muon_momentums_dict[var_key]

        if isinstance(gradient, tf.IndexedSlices):
            gradient = tf.scatter_nd(
                tf.expand_dims(gradient.indices, axis=-1),
                gradient.values,
                tf.shape(variable),
            )

        m.assign_add((gradient - m) * (1 - momentum))

        if self.nesterov:
            update = gradient + momentum * (m - gradient)
        else:
            update = m

        original_shape = tf.shape(update)
        rank = len(update.shape)
        if rank > 2:
            update = tf.reshape(update, [-1, update.shape[-1]])

        update = zeropower_via_newtonschulz5(update, steps=self.ns_steps)

        rows = tf.cast(tf.shape(update)[-2], update.dtype)
        cols = tf.cast(tf.shape(update)[-1], update.dtype)
        scale = tf.sqrt(tf.maximum(tf.cast(1.0, update.dtype), rows / cols))
        update = update * scale

        if rank > 2:
            update = tf.reshape(update, original_shape)

        if self.weight_decay != 0:
            variable.assign_sub(
                variable * tf.cast(self.lr * self.weight_decay, variable.dtype)
            )

        variable.assign_sub(update * lr)

    def _adamw_update_step(self, gradient, variable, var_key):
        lr = tf.cast(self.adamw_lr, variable.dtype)
        if hasattr(variable, "lr_multiplier"):
            lr = lr * variable.lr_multiplier

        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(
            tf.cast(self.adamw_beta_1, variable.dtype), local_step
        )
        beta_2_power = tf.pow(
            tf.cast(self.adamw_beta_2, variable.dtype), local_step
        )

        m = self._adamw_momentums_dict[var_key]
        v = self._adamw_velocities_dict[var_key]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            m.assign_add(-m * (1 - self.adamw_beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.adamw_beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - self.adamw_beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - self.adamw_beta_2),
                    gradient.indices,
                )
            )
            if self.adamw_weight_decay != 0:
                variable.assign_sub(
                    variable
                    * tf.cast(
                        self.adamw_lr * self.adamw_weight_decay, variable.dtype
                    )
                )
            variable.assign_sub(
                (m * alpha) / (tf.sqrt(v) + self.adamw_epsilon)
            )
        else:
            m.assign_add((gradient - m) * (1 - self.adamw_beta_1))
            v.assign_add(
                (tf.square(gradient) - v) * (1 - self.adamw_beta_2)
            )
            if self.adamw_weight_decay != 0:
                variable.assign_sub(
                    variable
                    * tf.cast(
                        self.adamw_lr * self.adamw_weight_decay, variable.dtype
                    )
                )
            variable.assign_sub(
                (m * alpha) / (tf.sqrt(v) + self.adamw_epsilon)
            )

    def _clip_gradients(self, grads):
        clipped_grads = []
        for g in grads:
            g = replace_nan(g, tf.cast(0.0, g.dtype))
            clipped_grads.append(g)
        grads = super()._clip_gradients(clipped_grads)
        return grads

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
