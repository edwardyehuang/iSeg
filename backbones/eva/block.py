# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

import tensorflow as tf

from iseg.backbones.eva.attention import EvaAttention

from iseg.backbones.eva.swiglu import SwiGLU
from iseg.backbones.eva.glumlp import GluMlp
from iseg.backbones.eva.mlp import Mlp

from iseg.utils.drops import drop_path

LAYER_NORM_EPSILON = 1e-6

class EvaBlock (tf.keras.Model):

    def __init__(
        self,
        num_heads=8,
        qkv_bias=True,
        qkv_fused=True,
        mlp_ratio=4.0,
        swiglu_mlp=False,
        scale_mlp=False,
        scale_attention_inner=False,
        attention_dropout_rate=0.0,
        projection_dropout_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        activation=tf.nn.gelu,
        attention_head_filters=None,
        use_post_norm=False,
        trainable=True,
        name=None,
    ):
        
        super().__init__(trainable=trainable, name=name)

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qkv_fused = qkv_fused

        self.mlp_ratio = mlp_ratio
        self.swiglu_mlp = swiglu_mlp
        self.scale_mlp = scale_mlp

        self.scale_attention_inner = scale_attention_inner

        self.attention_dropout_rate = attention_dropout_rate
        self.projection_dropout_rate = projection_dropout_rate
        self.drop_path_rate = drop_path_rate

        self.init_values = init_values

        self.activation = activation

        self.attention_head_filters = attention_head_filters

        self.use_post_norm = use_post_norm


    def build(self, input_shape):

        input_channels = input_shape[0][-1]

        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NORM_EPSILON,
            name=f"{self.name}/norm1",
        )

        self.attention = EvaAttention(
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qkv_fused=self.qkv_fused,
            attention_dropout_rate=self.attention_dropout_rate,
            projection_dropout_rate=self.projection_dropout_rate,
            attention_head_filters=self.attention_head_filters,
            use_norm=self.scale_attention_inner,
            name=f"{self.name}/attn",
        )

        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=LAYER_NORM_EPSILON,
            name=f"{self.name}/norm2",
        )
        
        hidden_channels = int(input_channels * self.mlp_ratio)

        mlp_name = f"{self.name}/mlp"

        if self.swiglu_mlp:
            if self.scale_mlp:
                self.mlp = SwiGLU(
                    hidden_filters=hidden_channels,
                    use_norm=True,
                    activation=self.activation,
                    dropout_rate=self.projection_dropout_rate,
                    name=mlp_name,
                )
            else:
                self.mlp = GluMlp(
                    hidden_filters=hidden_channels * 2,
                    use_norm=False,
                    activation=tf.nn.silu,
                    dropout_rate=self.projection_dropout_rate,
                    name=mlp_name,
                )
        else:
            self.mlp = Mlp(
                hidden_filters=hidden_channels,
                activation=self.activation,
                use_norm=self.scale_mlp,
                dropout_rate=self.projection_dropout_rate,
                name=mlp_name,
            )

        if self.init_values is not None:           
            self.gamma_1 = self.add_weight(
                name="gamma_1",
                shape=[input_channels],
                initializer=tf.keras.initializers.Constant(self.init_values),
                trainable=True,
            )

            self.gamma_2 = self.add_weight(
                name="gamma_2",
                shape=[input_channels],
                initializer=tf.keras.initializers.Constant(self.init_values),
                trainable=True,
            )
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0

        super().build(input_shape)

    
    def call (self, inputs, training=None):

        x = inputs[0]
        rope = inputs[1]

        residual = tf.identity(x, name="residual")

        # attention

        if not self.use_post_norm:
            x = self.norm1(x)

        x = self.attention([x, rope], training=training)

        if self.use_post_norm:
            x = self.norm1(x)

        x *= self.gamma_1
        x = drop_path(x, self.drop_path_rate, training=training)

        x += tf.cast(residual, x.dtype)
        residual = tf.identity(x, name="residual_2")

        # mlp

        if not self.use_post_norm:
            x = self.norm2(x)

        x = self.mlp(x, training=training)

        if self.use_post_norm:
            x = self.norm2(x)

        x *= self.gamma_2
        x += residual

        return x

