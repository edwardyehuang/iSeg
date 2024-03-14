# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import tensorflow as tf
import keras

from keras_nlp.src.utils.keras_utils import clone_initializer
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from iseg.utils.common import get_tensor_shape


class CachedGemmaAttention(keras.layers.Layer):
    """A cached grouped query attention layer."""

    def __init__(
        self,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        kernel_initializer="glorot_uniform",
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dropout = dropout

        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )
        self.num_key_value_groups = num_query_heads // num_key_value_heads

    def build(self, inputs_shape):
        self.hidden_dim = inputs_shape[-1]

        self.query_dense = keras.layers.EinsumDense(
            "btd,ndh->btnh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            "bsd,kdh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            "bsd,kdh->bskh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(inputs_shape)

        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
        )

        self.output_dense = keras.layers.EinsumDense(
            equation="btnh,nhd->btd",
            output_shape=(None, self.hidden_dim),
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )
        self.softmax = keras.layers.Softmax(dtype=tf.float32)
        self.built = True

    @tf.autograph.experimental.do_not_convert
    def _apply_rope(self, x, positions):
        """Rope rotate q or k."""
        # TODO: refactor to use RotaryEmbedding layer?
        max_wavelength = 10000
        x_shape = tf.shape(x)
        freq_exponents = (2.0 / tf.cast(x_shape[-1], dtype=self.compute_dtype)) * tf.cast(
            tf.range(x_shape[-1] // 2, dtype=tf.int32), self.compute_dtype
        )
        timescale = max_wavelength**freq_exponents
        radians = positions[..., None] / timescale[None, None, :]
        radians = radians[..., None, :]
        sin, cos = tf.sin(radians), tf.cos(radians)
        x1, x2 = tf.split(x, 2, axis=-1)
        # Avoid `ops.concatenate` for now, to avoid a obscure bug with XLA
        # compilation on jax. We should be able to remove this once the
        # following PR is in all jax releases we care about:
        # https://github.com/openxla/xla/pull/7875
        output = tf.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
        return tf.reshape(output, x_shape)

    def _compute_attention(
        self,
        q,
        k,
        v,
        attention_mask,
        training=False,
    ):
        query_normalization = 1 / np.sqrt(self.head_dim)

        q *= tf.cast(query_normalization, dtype=q.dtype)
        q_shape = get_tensor_shape(q, return_list=True)
        q = tf.reshape(
            q,
            (
                *q_shape[:-2],
                self.num_key_value_heads,
                self.num_query_heads // self.num_key_value_heads,
                q_shape[-1],
            ),
        )
        b, q_len, _, _, h = get_tensor_shape(q)

        attention_logits = tf.einsum("btkgh,bskh->bkgts", q, k)
        attention_mask = attention_mask[:, None, None, :, :]
        orig_dtype = attention_logits.dtype
        attention_softmax = self.softmax(attention_logits, mask=attention_mask)
        attention_softmax = tf.cast(attention_softmax, orig_dtype)

        if self.dropout:
            attention_softmax = self.dropout_layer(
                attention_softmax, training=training
            )

        results = tf.einsum("bkgts,bskh->btkgh", attention_softmax, v)
        return tf.reshape(results, (b, q_len, self.num_query_heads, h))

    def call(
        self,
        x,
        attention_mask=None,
        cache=None,
        cache_update_index=0,
        training=False,
    ):
        seq_len = tf.shape(x)[1]
        start_index = cache_update_index
        positions = tf.cast(
            tf.range(seq_len, dtype="float32"), self.compute_dtype
        )
        positions = positions + tf.cast(start_index, self.compute_dtype)
        query = self.query_dense(x)
        query = self._apply_rope(query, positions)

        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            key_update = self.key_dense(x)
            key_update = self._apply_rope(key_update, positions)
            value_update = self.value_dense(x)
            start = [0, cache_update_index, 0, 0]
            key = dynamic_update_slice(key_cache, key_update, start)
            value = dynamic_update_slice(value_cache, value_update, start)
            cache = tf.stack((key, value), axis=1)
        else:
            key = self.key_dense(x)
            key = self._apply_rope(key, positions)
            value = self.value_dense(x)

        attention_vec = self._compute_attention(
            query, key, value, attention_mask, training=training
        )

        # Wipe attn vec if there are no attended tokens.
        no_attended_tokens = tf.reduce_all(
            tf.equal(attention_mask, False), axis=-1, keepdims=True
        )[..., None]
        attention_vec = tf.where(
            no_attended_tokens, tf.zeros_like(attention_vec), attention_vec
        )

        attention_output = self.output_dense(attention_vec)

        if cache is not None:
            return attention_output, cache
        return attention_output
