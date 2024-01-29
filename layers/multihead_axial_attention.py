import tensorflow as tf

from iseg.initializers.shared_initializers import SharedInitializer
from iseg.layers.model_builder import get_tensor_shape_v2
from iseg.layers.normalizations import normalization
from iseg.utils.keras_ops import replace_nan
from iseg import check_numerics

def safed_softmax (x):
    t = x.dtype
    x = tf.cast(x, tf.float32)
    x = tf.nn.softmax(x)
    x = tf.cast(x, t)

    return x


class MultiHeadAxialAttentionLayer (tf.keras.Model):

    def __init__(
        self, 
        filters=-1,
        num_heads=4,
        apply_linear=True,
        apply_scale=True,
        shared_qk_weights=True,
        shared_qk=False,
        trainable=True,
        linear_func=tf.keras.layers.Conv2D,
        name=None
    ):

        super().__init__(trainable=trainable, name=name)

        self.filters = filters
        self.num_heads = num_heads
        self.apply_linear = apply_linear
        self.apply_scale = apply_scale
        self.shared_qk_weights = shared_qk_weights

        self.shared_qk = shared_qk

        self.linear_func = linear_func

    
    def build (self, input_shape):

        channels = input_shape[-1]

        qk_filters = channels if self.filters == -1 else self.filters

        if self.shared_qk_weights:
            q_kernel_initializer = k_kernel_initializer = SharedInitializer(tf.keras.initializers.GlorotUniform())
        else:
            q_kernel_initializer = tf.keras.initializers.GlorotUniform()
            k_kernel_initializer = tf.keras.initializers.GlorotUniform()

        if self.apply_linear:
            self.query_conv = self.linear_func(
                qk_filters,
                (1, 1), 
                kernel_initializer=q_kernel_initializer,
                trainable=self.trainable,
                name="query_conv"
                )
            
            if not self.shared_qk:  
                self.key_conv = self.linear_func(
                    qk_filters, 
                    (1, 1), 
                    kernel_initializer=k_kernel_initializer,
                    trainable=self.trainable,
                    name="key_conv"
                )

            self.value_conv = self.linear_func(
                channels, 
                (1, 1), 
                trainable=self.trainable,
                name="value_conv"
            )


    def compute_attetnion (self, query, key, value, training=None):

        batch_size, height, width, _ = get_tensor_shape_v2(value)

        query = check_numerics(query, "query contains NaN/Inf", level=1)
        key = check_numerics(key, "keys contains NaN/Inf", level=1)

        v_query = tf.reshape(query, [batch_size, height, width, self.num_heads, query.shape[-1] // self.num_heads]) # [N, H, W, heads, C]
        v_query = tf.transpose(v_query,(0, 3, 2, 1, 4)) # [N, heads, W, H, C]
        v_key = tf.reshape(key, [batch_size, height, width, self.num_heads, key.shape[-1] // self.num_heads])
        v_key = tf.transpose(v_key,(0, 3, 2, 4, 1)) # [N, heads, W, C, H]
        v_attention_map = tf.matmul(v_query, v_key) # [N, heads, W, H, H]

        u_query = tf.reshape(query, [batch_size, height, width, self.num_heads, query.shape[-1] // self.num_heads]) # [N, H, W, heads, C]
        u_query = tf.transpose(u_query,(0, 3, 1, 2, 4))
        u_key = tf.reshape(key, [batch_size, height, width, self.num_heads, key.shape[-1] // self.num_heads]) # [N, H, W, heads, C]
        u_key = tf.transpose(u_key,(0, 3, 1, 4, 2)) # [N, heads, H, C, W]
        u_attention_map = tf.matmul(u_query, u_key) # [N, heads, H, W, W]

        v_attention_map = tf.clip_by_value(v_attention_map, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        u_attention_map = tf.clip_by_value(u_attention_map, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        v_attention_map = check_numerics(v_attention_map, "v_attention_map contains NaN/Inf", level=1)
        u_attention_map = check_numerics(u_attention_map, "u_attention_map contains NaN/Inf", level=1)

        if self.apply_scale:
            v_attention_map /= tf.sqrt(tf.cast(v_query.shape[-1], v_attention_map.dtype))
            u_attention_map /= tf.sqrt(tf.cast(u_query.shape[-1], u_attention_map.dtype))

        v_attention_map = safed_softmax(v_attention_map) # [N, heads, W, H, H]
        u_attention_map = safed_softmax(u_attention_map) # [N, heads, H, W, W]

        v_attention_map = replace_nan(v_attention_map, tf.keras.backend.epsilon())
        u_attention_map = replace_nan(u_attention_map, tf.keras.backend.epsilon())

        v_attention_map = tf.clip_by_value(v_attention_map, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        u_attention_map = tf.clip_by_value(u_attention_map, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())

        v_attention_map = check_numerics(v_attention_map, "softmax v_attention_map contains NaN/Inf", level=1)
        u_attention_map = check_numerics(u_attention_map, "softmax u_attention_map contains NaN/Inf", level=1)

        x = tf.reshape(value, [batch_size, height, width, self.num_heads, value.shape[-1] // self.num_heads])
        x = tf.transpose(x,(0, 3, 2, 1, 4)) # [N, heads, W, H, C]
        x = tf.matmul(v_attention_map, x)  # [N, heads, W, H, C]

        x = tf.transpose(x, [0, 1, 3, 2, 4])  # [N, heads, H, W, C]
        x = tf.matmul(u_attention_map, x) # [N, heads, H, W, C]

        # if self.apply_norm:
            # x = self.final_bn(x, training=training)

        x = tf.transpose(x, [0, 2, 3, 4, 1]) # [N, H, W, C, heads]

        x = tf.reshape(x, [batch_size, height, width, x.shape[-2] * self.num_heads])

        return x


    def call (self, inputs, training=None):

        x = inputs # [N, H, W, C]

        if self.apply_linear:
            query = self.query_conv(x) # [N, H, W, C]

            if not self.shared_qk:
                key = self.key_conv(x)
            else:
                key = tf.identity(query, name="shared_key")

            x = self.value_conv(x) # [N, H, W, C]
            
        else:
            query = x
            key = x


        x = self.compute_attetnion(query, key, x, training=training)

        return x