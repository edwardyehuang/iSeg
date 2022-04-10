# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


class WarmUpPolyDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        end_learning_rate=0.0001,
        warmup_steps=0,
        warmup_learning_rate=1e-4,
        power=1.0,
        name=None,
    ):
        """Applies a warmup polynomial decay to the learning rate.
        Args:
        initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
        decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  See the decay computation above.
        end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The minimal end learning rate.
        power: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The power of the polynomial. Defaults to linear, 1.0.
        cycle: A boolean, whether or not it should cycle beyond decay_steps.
        name: String.  Optional name of the operation. Defaults to
            'PolynomialDecay'.
        """
        super(WarmUpPolyDecay, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.warmup_steps = warmup_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUpPolyDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            end_learning_rate = tf.cast(self.end_learning_rate, dtype)
            power = tf.cast(self.power, dtype)

            current_step = tf.cast(step, dtype)
            max_steps = tf.cast(self.decay_steps, dtype) - self.warmup_steps

            current_step = tf.math.minimum(current_step, max_steps)

            adjusted_slow_start_learning_rate = self.warmup_learning_rate
            adjusted_current_step = current_step

            if self.warmup_steps > 0:
                adjusted_current_step = tf.maximum(adjusted_current_step - self.warmup_steps, 0)
                adjusted_slow_start_learning_rate = (
                    self.warmup_learning_rate
                    + (initial_learning_rate - self.warmup_learning_rate)
                    * tf.cast(current_step, tf.float32)
                    / self.warmup_steps
                )

            p = tf.math.divide(adjusted_current_step, max_steps)

            learning_rate = tf.add(
                tf.math.multiply(initial_learning_rate - end_learning_rate, tf.math.pow(1 - p, power)),
                end_learning_rate,
                name=name,
            )

            return tf.cond(step < self.warmup_steps, lambda: adjusted_slow_start_learning_rate, lambda: learning_rate)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "warmup_steps": self.warmup_steps,
            "warmup_learning_rate": self.warmup_learning_rate,
            "name": self.name,
        }


if __name__ == "__main__":
    
    d = WarmUpPolyDecay(1e-2, 30000, end_learning_rate=0, warmup_steps=1500, warmup_learning_rate=0)
    
    print(d(0))
    print(d(500))
    print(d(1000))
    print(d(1500))
    print(d(2000))
