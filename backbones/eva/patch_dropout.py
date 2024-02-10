# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

import tensorflow as tf

from iseg.utils import get_tensor_shape


class PatchDropout (tf.keras.layers.Layer):

    def __init__(
        self,
        prob=0.5,
        num_prefix_tokens=1,
        ordered=True,
        return_indices=False,
        name=None
    ):
        super().__init__(name=name)

        assert 0 <= prob <= 1

        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens # exclude class token (or other prefix tokens)
        self.ordered = ordered
        self.return_indices = return_indices


    def call (self, inputs, training=None):

        x = inputs

        if not training or self.prob == 0.:
            if self.return_indices:
                return x, None
            
            return x

        
        if self.num_prefix_tokens > 0:
            prefix_tokens = x[:, :self.num_prefix_tokens, ...]
            x = x[:, self.num_prefix_tokens:, ...]
        else:
            prefix_tokens = None

        batch_size, num_patches, _ = get_tensor_shape(x)

        num_keep = max(1, int(num_patches * (1. - self.prob)))
        rand = tf.random.normal([batch_size, num_patches])
        indices = tf.argsort(rand, axis=-1, direction='ASCENDING')
        keep_indices = indices[:, :num_keep] # [batch_size, num_keep]

        if self.ordered:
            keep_indices = tf.sort(keep_indices, axis=-1) # [batch_size, num_keep]

        x = tf.gather(x, keep_indices, axis=1) # [batch_size, num_keep, channels]

        if prefix_tokens is not None:
            x = tf.concat([prefix_tokens, x], axis=1) # [batch_size, num_keep + num_prefix_tokens, channels]

        if self.return_indices:
            return x, keep_indices
        
        return x






    

