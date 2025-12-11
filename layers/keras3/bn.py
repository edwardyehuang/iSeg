import tensorflow as tf
import keras
from keras.src import backend
from keras.src import ops
from keras.src.backend.tensorflow.core import cast

from iseg.distribution.distribution_utils import all_reduce_values


def moments(x, axes, keepdims=False, synchronized=False):
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = keras.backend.standardize_dtype(x.dtype)
    if ori_dtype in ("float16", "bfloat16"):
        need_cast = True
        x = cast(x, "float32")

    mean, variance = _compute_moments_sync(
        x, axes, keepdims, synchronized
    )

    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = tf.clip_by_value(mean, tf.float16.min, tf.float16.max)
        variance = tf.clip_by_value(variance, tf.float16.min, tf.float16.max)
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def _compute_moments_sync(x, axes, keepdims, synchronized=True):
    replica_ctx : tf.distribute.ReplicaContext = tf.distribute.get_replica_context()

    with tf.name_scope("moments"):
        if replica_ctx and synchronized:
            local_sum = tf.reduce_sum(x, axis=axes, keepdims=True)
            local_squared_sum = tf.reduce_sum(tf.square(x), axis=axes, keepdims=True)
            batch_size = tf.cast(tf.shape(x)[axes[0]], tf.int32)

            # TODO(b/163099951): batch the all-reduces once we sort out the
            # ordering issue for NCCL. We don't have a mechanism to launch
            # NCCL in the same order in each replica nowadays, so we limit
            # NCCL to batch all-reduces.
            y_sum = all_reduce_values(local_sum)
            y_squared_sum = all_reduce_values(local_squared_sum)
            global_batch_size = all_reduce_values(batch_size)

            axes_vals = [(tf.shape(x))[axes[i]] for i in range(1, len(axes))]
            multiplier = tf.cast(tf.reduce_prod(axes_vals), tf.int32)
            multiplier = tf.cast(multiplier * global_batch_size, tf.float32)

            mean = tf.math.divide_no_nan(y_sum, multiplier)
            y_squared_mean = tf.math.divide_no_nan(y_squared_sum, multiplier)
            # var = E(x^2) - E(x)^2
            variance = y_squared_mean - tf.square(mean)
        else:
            # Compute true mean while keeping the dims for proper broadcasting.
            mean = tf.reduce_mean(x, axes, keepdims=True, name="mean")
            # sample variance, not unbiased variance
            # Note: stop_gradient does not change the gradient that gets
            #       backpropagated to the mean from the variance calculation,
            #       because that gradient is zero
            variance = tf.reduce_mean(
                tf.math.squared_difference(x, tf.stop_gradient(mean)), axes, keepdims=True, name="variance"
            )

        if not keepdims:
            mean = tf.squeeze(mean, axes)
            variance = tf.squeeze(variance, axes)

    return mean, variance


class BatchNormalization_Patch (keras.layers.BatchNormalization):

    def _moments(self, inputs, mask):
        if mask is None:
            return moments(
                inputs,
                axes=self._reduction_axes,
                synchronized=self.synchronized,
            )

        mask_weights = ops.cast(
            mask,
            inputs.dtype,
        )
        mask_weights_broadcasted = ops.expand_dims(
            mask_weights,
            axis=-1,
        )
        weighted_inputs = mask_weights_broadcasted * inputs

        weighted_input_sum = ops.sum(
            weighted_inputs,
            self._reduction_axes,
            keepdims=True,
        )
        sum_of_weights = ops.sum(
            mask_weights_broadcasted,
            self._reduction_axes,
            keepdims=True,
        )
        mean = weighted_input_sum / (sum_of_weights + backend.config.epsilon())

        difference = weighted_inputs - mean
        squared_difference = ops.square(difference)
        weighted_distsq = ops.sum(
            mask_weights_broadcasted * squared_difference,
            self._reduction_axes,
            keepdims=True,
        )
        variance = weighted_distsq / (sum_of_weights + backend.config.epsilon())

        return ops.squeeze(mean), ops.squeeze(variance)
    


    


