# Copied from TensorFlow Addon repo

import tensorflow as tf


class DecoupledWeightDecayExtension:
    """This class allows to extend optimizers with decoupled weight decay.
    It implements the decoupled weight decay described by Loshchilov & Hutter
    (https://arxiv.org/pdf/1711.05101.pdf), in which the weight decay is
    decoupled from the optimization steps w.r.t. to the loss function.
    For SGD variants, this simplifies hyperparameter search since it decouples
    the settings of weight decay and learning rate.
    For adaptive gradient algorithms, it regularizes variables with large
    gradients more than L2 regularization would, which was shown to yield
    better training loss and generalization error in the paper above.
    This class alone is not an optimizer but rather extends existing
    optimizers with decoupled weight decay. We explicitly define the two
    examples used in the above paper (SGDW and AdamW), but in general this can
    extend any OptimizerX class by using
        `ExtendedCls = extend_with_decoupled_weight_decay(OptimizerX)`.
    Weight decay can then be set when instantiating the optimizer:
        `optimizerX = ExtendedCls(weight_decay=0.001, learning_rate=0.001)`.
    In order for it to work, it must be the first class the Optimizer with
    weight decay inherits from, e.g.
    ```python
    class AdamW(DecoupledWeightDecayExtension, tf.keras.optimizers.Adam):
      def __init__(self, weight_decay, *args, **kwargs):
        super(AdamW, self).__init__(weight_decay, *args, **kwargs).
    ```
    Note: this extension decays weights BEFORE applying the update based
    on the gradient, i.e. this extension only has the desired behaviour for
    optimizers which do not depend on the value of'var' in the update step!
    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:
    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)
    # ...
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```
    """

    def __init__(self, weight_decay, **kwargs):
        """Extension class that adds weight decay to an optimizer.
        Args:
            weight_decay: A `Tensor`, a floating point value, or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                to decay the variable by, in the update step.
            **kwargs: Optional list or tuple or set of `Variable` objects to
                decay.
        """
        wd = kwargs.pop("weight_decay", weight_decay)
        super().__init__(**kwargs)
        self._decay_var_list = None  # is set in minimize or apply_gradients
        self._set_hyper("weight_decay", wd)

    def get_config(self):
        config = super().get_config()
        config.update({"weight_decay": self._serialize_hyperparameter("weight_decay")})
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # LR handling copied from optimizer_v2.OptimizerV2
        if "learning_rate" in config:
            if isinstance(config["learning_rate"], dict):
                config["learning_rate"] = tf.keras.optimizers.schedules.deserialize(
                    config["learning_rate"], custom_objects=custom_objects
                )

        if "weight_decay" in config:
            if isinstance(config["weight_decay"], dict):
                config["weight_decay"] = tf.keras.optimizers.schedules.deserialize(
                    config["weight_decay"], custom_objects=custom_objects
                )

        return cls(**config)

    def minimize(self, loss, var_list, grad_loss=None, name=None, decay_var_list=None, tape=None):
        """Minimize `loss` by updating `var_list`.
        This method simply computes gradient using `tf.GradientTape` and calls
        `apply_gradients()`. If you want to process the gradient before
        applying then call `tf.GradientTape` and `apply_gradients()` explicitly
        instead of using this function.
        Args:
            loss: `Tensor` or callable. If a callable, `loss` should take no
                arguments and return the value to minimize. If a `Tensor`, the
                `tape` argument must be passed.
            var_list: list or tuple of `Variable` objects to update to
                minimize `loss`, or a callable returning the list or tuple of
                `Variable` objects. Use callable when the variable list would
                otherwise be incomplete before `minimize` since the variables
                are created at the first time `loss` is called.
            grad_loss: Optional. A `Tensor` holding the gradient computed for
                `loss`.
            decay_var_list: Optional list of variables to be decayed. Defaults
                to all variables in var_list.
            name: Optional name for the returned operation.
            tape: (Optional) `tf.GradientTape`. If `loss` is provided as a
                `Tensor`, the tape that computed the `loss` must be provided.
        Returns:
            An Operation that updates the variables in `var_list`.
        Raises:
            ValueError: If some of the variables are not `Variable` objects.
        """
        self._decay_var_list = set([v.ref() for v in decay_var_list]) if decay_var_list else False
        return super().minimize(loss, var_list=var_list, grad_loss=grad_loss, name=name, tape=tape)

    def apply_gradients(self, grads_and_vars, name=None, decay_var_list=None, **kwargs):
        """Apply gradients to variables.
        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.
        Args:
            grads_and_vars: List of (gradient, variable) pairs.
            name: Optional name for the returned operation.  Default to the
                name passed to the `Optimizer` constructor.
            decay_var_list: Optional list of variables to be decayed. Defaults
                to all variables in var_list.
            **kwargs: Additional arguments to pass to the base optimizer's
                apply_gradient method, e.g., TF2.2 added an argument
                `experimental_aggregate_gradients`.
        Returns:
            An `Operation` that applies the specified gradients.
        Raises:
            TypeError: If `grads_and_vars` is malformed.
            ValueError: If none of the variables have gradients.
        """
        self._decay_var_list = set([v.ref() for v in decay_var_list]) if decay_var_list else False
        return super().apply_gradients(grads_and_vars, name=name, **kwargs)

    def _decay_weights_op(self, var, apply_state=None):
        if not self._decay_var_list or var.ref() in self._decay_var_list:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(
                var_device, var_dtype
            )

            return var.assign_sub(coefficients["wd_t"] * var, self._use_locking)
        return tf.no_op()

    def _decay_weights_sparse_op(self, var, indices, apply_state=None):
        if not self._decay_var_list or var.ref() in self._decay_var_list:
            var_device, var_dtype = var.device, var.dtype.base_dtype
            coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(
                var_device, var_dtype
            )

            update = -coefficients["wd_t"] * tf.gather(var, indices)
            return self._resource_scatter_add(var, indices, update)
        return tf.no_op()

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(DecoupledWeightDecayExtension, self)._prepare_local(var_device, var_dtype, apply_state)

        if "weight_decay" in self._hyper:
            wd_t = tf.identity(self._decayed_wd(var_dtype))
            apply_state[(var_device, var_dtype)]["wd_t"] = wd_t

    def _decayed_wd(self, var_dtype):
        wd_t = self._get_hyper("weight_decay", var_dtype)

        if isinstance(wd_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            wd_t = tf.cast(wd_t(self.iterations), var_dtype)

        return wd_t

    # Here, we overwrite the apply functions that the base optimizer calls.
    # super().apply_x resolves to the apply_x function of the BaseOptimizer.

    def _resource_apply_dense(self, grad, var, apply_state=None):
        with tf.control_dependencies([self._decay_weights_op(var, apply_state=apply_state)]):
            return super()._resource_apply_dense(grad, var, apply_state=apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        decay_op = self._decay_weights_sparse_op(var, indices, apply_state=apply_state)
        with tf.control_dependencies([decay_op]):
            return super()._resource_apply_sparse(grad, var, indices, apply_state=apply_state)


class AdamW(DecoupledWeightDecayExtension, tf.keras.optimizers.Adam):
    """Optimizer that implements the Adam algorithm with weight decay.
    This is an implementation of the AdamW optimizer described in "Decoupled
    Weight Decay Regularization" by Loshch ilov & Hutter
    (https://arxiv.org/abs/1711.05101)
    ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).
    It computes the update step of `tf.keras.optimizers.Adam` and additionally
    decays the variable. Note that this is different from adding L2
    regularization on the variables to the loss: it regularizes variables with
    large gradients more than L2 regularization would, which was shown to yield
    better training loss and generalization error in the paper above.
    For further information see the documentation of the Adam Optimizer.
    This optimizer can also be instantiated as
    ```python
    extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam,
                                       weight_decay=weight_decay)
    ```
    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:
    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)
    # ...
    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```
    """

    def __init__(
        self,
        weight_decay,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad: bool = False,
        name: str = "AdamW",
        **kwargs,
    ):
        """Construct a new AdamW optimizer.
        For further information see the documentation of the Adam Optimizer.
        Args:
            weight_decay: A Tensor or a floating point value. The weight decay.
            learning_rate: A Tensor or a floating point value. The learning
                rate.
            beta_1: A float value or a constant float tensor. The exponential
                decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor. The exponential
                decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability. This epsilon is
                "epsilon hat" in the Kingma and Ba paper (in the formula just
                before Section 2.1), not the epsilon in Algorithm 1 of the
                paper.
            amsgrad: boolean. Whether to apply AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                beyond".
            name: Optional name for the operations created when applying
                gradients. Defaults to "AdamW".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse decay
                of learning rate. `lr` is included for backward compatibility,
                recommended to use `learning_rate` instead.
        """
        super().__init__(
            weight_decay,
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs,
        )
