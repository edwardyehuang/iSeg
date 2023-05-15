# Copied from TensorFlow 2.12

import math
import tensorflow as tf


class CosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a cosine decay with optional warmup.

    See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    SGDR: Stochastic Gradient Descent with Warm Restarts.

    For the idea of a linear warmup of our learning rate,
    see [Goyal et al.](https://arxiv.org/pdf/1706.02677.pdf).

    When we begin training a model, we often want an initial increase in our
    learning rate followed by a decay. If `warmup_target` is an int, this
    schedule applies a linear increase per optimizer step to our learning rate
    from `initial_learning_rate` to `warmup_target` for a duration of
    `warmup_steps`. Afterwards, it applies a cosine decay function taking our
    learning rate from `warmup_target` to `alpha` for a duration of
    `decay_steps`. If `warmup_target` is None we skip warmup and our decay
    will take our learning rate from `initial_learning_rate` to `alpha`.
    It requires a `step` value to  compute the learning rate. You can
    just pass a TensorFlow variable that you increment at each training step.

    The schedule is a 1-arg callable that produces a warmup followed by a
    decayed learning rate when passed the current optimizer step. This can be
    useful for changing the learning rate value across different invocations of
    optimizer functions.

    Our warmup is computed as:

    ```python
    def warmup_learning_rate(step):
        completed_fraction = step / warmup_steps
        total_delta = target_warmup - initial_learning_rate
        return completed_fraction * total_delta
    ```

    And our decay is computed as:

    ```python
    if warmup_target is None:
        initial_decay_lr = initial_learning_rate
    else:
        initial_decay_lr = warmup_target

    def decayed_learning_rate(step):
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_decay_lr * decayed
    ```

    Example usage without warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0.1
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps)
    ```

    Example usage with warmup:

    ```python
    decay_steps = 1000
    initial_learning_rate = 0
    warmup_steps = 1000
    target_learning_rate = 0.1
    lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
    )
    ```

    You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
    as the learning rate. The learning rate schedule is also serializable and
    deserializable using `tf.keras.optimizers.schedules.serialize` and
    `tf.keras.optimizers.schedules.deserialize`.

    Returns:
      A 1-arg callable learning rate schedule that takes the current optimizer
      step and outputs the decayed learning rate, a scalar `Tensor` of the same
      type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        alpha=0.0,
        name=None,
        warmup_target=None,
        warmup_steps=0,
    ):
        """Applies cosine decay to the learning rate.

        Args:
          initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python int. The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python int.
            Number of steps to decay over.
          alpha: A scalar `float32` or `float64` `Tensor` or a Python int.
            Minimum learning rate value for decay as a fraction of
            `initial_learning_rate`.
          name: String. Optional name of the operation.  Defaults to
            'CosineDecay'.
          warmup_target: None or a scalar `float32` or `float64` `Tensor` or a
            Python int. The target learning rate for our warmup phase. Will cast
            to the `initial_learning_rate` datatype. Setting to None will skip
            warmup and begins decay phase from `initial_learning_rate`.
            Otherwise scheduler will warmup from `initial_learning_rate` to
            `warmup_target`.
          warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python int.
            Number of steps to warmup over.
        """
        super().__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        self.warmup_steps = warmup_steps
        self.warmup_target = warmup_target

    def _decay_function(self, step, decay_steps, decay_from_lr, dtype):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / decay_steps
            tf_pi = tf.constant(math.pi, dtype=dtype)
            cosine_decayed = 0.5 * (1.0 + tf.cos(tf_pi * completed_fraction))
            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return tf.multiply(decay_from_lr, decayed)

    def _warmup_function(
        self, step, warmup_steps, warmup_target, initial_learning_rate
    ):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / warmup_steps
            total_step_delta = warmup_target - initial_learning_rate
            return total_step_delta * completed_fraction + initial_learning_rate

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            global_step_recomp = tf.cast(step, dtype)

            if self.warmup_target is None:
                global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
                return self._decay_function(
                    global_step_recomp,
                    decay_steps,
                    initial_learning_rate,
                    dtype,
                )

            warmup_target = tf.cast(self.warmup_target, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)

            global_step_recomp = tf.minimum(
                global_step_recomp, decay_steps + warmup_steps
            )

            return tf.cond(
                global_step_recomp < warmup_steps,
                lambda: self._warmup_function(
                    global_step_recomp,
                    warmup_steps,
                    warmup_target,
                    initial_learning_rate,
                ),
                lambda: self._decay_function(
                    global_step_recomp - warmup_steps,
                    decay_steps,
                    warmup_target,
                    dtype,
                ),
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "alpha": self.alpha,
            "name": self.name,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
        }
