# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

# This code is motified from https://github.com/keras-team/keras/blob/master/keras/layers/normalization/batch_normalization.py

import tensorflow as tf

from tensorflow.python.distribute import distribution_strategy_context as ds
from tensorflow.python.distribute import reduce_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from distutils.version import LooseVersion


class SyncBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        trainable=True,
        adjustment=None,
        name=None,
        **kwargs
    ):

        # Currently we only support aggregating over the global batch size.
        super(SyncBatchNormalization, self).__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            fused=False,
            trainable=trainable,
            virtual_batch_size=None,
            name=name,
            **kwargs
        )

    def _calculate_mean_and_var(self, x, axes, keep_dims):

        with ops.name_scope("moments", values=[x, axes]):
            # The dynamic range of fp16 is too limited to support the collection of
            # sufficient statistics. As a workaround we simply perform the operations
            # on 32-bit floats before converting the mean and variance back to fp16

            y = math_ops.cast(x, dtypes.float32) if x.dtype == dtypes.float16 else x

            replica_ctx = ds.get_replica_context()
            if replica_ctx:
                local_sum = math_ops.reduce_sum(y, axis=axes, keepdims=True)
                local_squared_sum = math_ops.reduce_sum(math_ops.square(y), axis=axes, keepdims=True)
                batch_size = math_ops.cast(array_ops.shape_v2(y)[axes[0]], dtypes.float32)

                """
                y_sum, y_squared_sum, global_batch_size = (
                    replica_ctx.all_reduce(reduce_util.ReduceOp.SUM, [
                        local_sum, local_squared_sum, batch_size]))
                """

                y_sum = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM, local_sum)
                y_squared_sum = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM, local_squared_sum)
                global_batch_size = replica_ctx.all_reduce(reduce_util.ReduceOp.SUM, batch_size)

                axes_vals = [(array_ops.shape_v2(y))[axes[i]] for i in range(1, len(axes))]
                multiplier = math_ops.cast(math_ops.reduce_prod(axes_vals), dtypes.float32)
                multiplier = multiplier * global_batch_size

                mean = y_sum / multiplier
                y_squared_mean = y_squared_sum / multiplier
                # var = E(x^2) - E(x)^2
                variance = y_squared_mean - math_ops.square(mean)
            else:
                # Compute true mean while keeping the dims for proper broadcasting.
                mean = math_ops.reduce_mean(y, axes, keepdims=True, name="mean")
                # sample variance, not unbiased variance
                # Note: stop_gradient does not change the gradient that gets
                #       backpropagated to the mean from the variance calculation,
                #       because that gradient is zero
                variance = math_ops.reduce_mean(
                    math_ops.squared_difference(y, array_ops.stop_gradient(mean)), axes, keepdims=True, name="variance"
                )
            if not keep_dims:
                mean = array_ops.squeeze(mean, axes)
                variance = array_ops.squeeze(variance, axes)
            if x.dtype == dtypes.float16:
                return (math_ops.cast(mean, dtypes.float16), math_ops.cast(variance, dtypes.float16))
            else:
                return (mean, variance)
