# coding=utf-8
# Copyright 2022 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code is largely modified based on offical implementation at
# https://github.com/google-research/deeplab2/blob/main/model/pixel_encoder/moat.py

# We fixed many bugs and improved code structure for TensorFlow 2.0


import tensorflow as tf

from .moat_blocks import MBConvBlock
from .moat_blocks import MOATBlock

from iseg.layers.normalizations import normalization
from iseg.utils.keras3_utils import Keras3_Model_Wrapper


# This handles the invalid name scope of tf.keras.sequential
# used in stem layers.
_STEM_LAYER_NAME_SCOPE = 'stem'

_STRIDE_16_POSITION_EMBEDDING_SIZE = 14
_STRIDE_32_POSITION_EMBEDDING_SIZE = 7

DEFAULT_POS_EMB_SIZE=[
    None, None,
    _STRIDE_16_POSITION_EMBEDDING_SIZE,
    _STRIDE_32_POSITION_EMBEDDING_SIZE
]

class MOAT(Keras3_Model_Wrapper):
    """MOAT backbone."""

    def __init__(
        self, 
        stem_size,
        block_type_list,
        num_blocks,
        hidden_size,
        stage_stride=[2, 2, 2, 2],
        expansion_rate=4,
        se_ratio=0.25,
        head_size=32,
        window_size=[None, None, None, None],
        position_embedding_size=[
            None, None,
            _STRIDE_16_POSITION_EMBEDDING_SIZE,
            _STRIDE_32_POSITION_EMBEDDING_SIZE
        ],
        use_checkpointing_for_attention=False,
        global_attention_at_end_of_moat_stage=False,
        relative_position_embedding_type="2d_multi_head",
        ln_epsilon=1e-5,
        pool_size=2,
        survival_prob=None,
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        bias_initializer=tf.zeros_initializer,
        return_endpoints=False,
        name="moat",
    ):
        super().__init__(name=name)

        stage_number = len(block_type_list)

        if position_embedding_size is None:
            position_embedding_size = [None] * stage_number
            relative_position_embedding_type = None

        if (len(num_blocks) != stage_number or len(hidden_size) != stage_number):
            raise ValueError('The lengths of block_type, num_blocks and hidden_size ',
                'should be the same.')

        self.stem_size = stem_size
        self.block_type = block_type_list
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size

        self.stage_stride = stage_stride
        self.expansion_rate = expansion_rate
        self.se_ratio = se_ratio
        self.head_size = head_size

        self.window_size = window_size
        self.position_embedding_size = position_embedding_size

        self.use_checkpointing_for_attention = use_checkpointing_for_attention

        self.global_attention_at_end_of_moat_stage = global_attention_at_end_of_moat_stage
        self.relative_position_embedding_type = relative_position_embedding_type

        self.ln_epsilon = ln_epsilon
        self.pool_size = pool_size
        self.survival_prob = survival_prob

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.return_endpoints = return_endpoints

    def _build_stem(self):
        stem_layers = []
        for i in range(len(self.stem_size)):
            conv_layer = tf.keras.layers.Conv2D(
                filters=self.stem_size[i],
                kernel_size=3,
                strides=2 if i == 0 else 1,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                use_bias=True,
                name=f"{_STEM_LAYER_NAME_SCOPE}/conv_{i}"
            )
            stem_layers.append(conv_layer)
            if i < len(self.stem_size) - 1:
                stem_layers.append(self.norm_class(name=f"{_STEM_LAYER_NAME_SCOPE}/norm_{i}"))
                stem_layers.append(
                    tf.keras.layers.Activation(self.activation, name=f"{_STEM_LAYER_NAME_SCOPE}/act_{i}")
                )
        # The name scope of tf.keras.Sequential is invalid, see error handling
        # in the part of loading checkpoints in function get_model.
        self._stem = tf.keras.Sequential(
            layers=stem_layers,
            name='stem'
        )

    def _adjust_survival_rate(self, block_id, total_num_blocks):
        survival_prob = self.survival_prob
        if survival_prob is not None:
            drop_rate = 1.0 - survival_prob
            survival_prob = 1.0 - drop_rate * block_id / total_num_blocks

        return survival_prob


    def build(self, input_shape) -> None:
        norm_class = normalization
        self.norm_class = norm_class
        self.activation = tf.nn.gelu

        self._build_stem()

        self._blocks = []
        total_num_blocks = sum(self.num_blocks)

        for stage_id in range(len(self.block_type)):
            stage_blocks = []

            stage_num_blocks = self.num_blocks[stage_id]
            stage_block_type = self.block_type[stage_id]

            for local_block_id in range(stage_num_blocks):

                block_stride = 1
                if local_block_id == 0:
                    block_stride = self.stage_stride[stage_id]

                block_id = sum(self.num_blocks[:stage_id]) + local_block_id
                survival_prob = self._adjust_survival_rate(
                    block_id, total_num_blocks
                )

                block_name = "block_{:0>2d}_{:0>2d}".format(stage_id, local_block_id)

                local_window_size = self.window_size[stage_id]

                if (local_block_id == stage_num_blocks - 1 and 
                    stage_block_type == "moat" and 
                    self.global_attention_at_end_of_moat_stage):

                    local_window_size = None

                if stage_block_type == "mbconv":
                    block = MBConvBlock(
                        hidden_size=self.hidden_size[stage_id],
                        expansion_rate=self.expansion_rate,
                        se_ratio=self.se_ratio,
                        block_stride=block_stride,
                        pool_size=self.pool_size,
                        norm_class=self.norm_class,
                        activation=self.activation,
                        survival_prob=survival_prob,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        name=block_name,
                    )
                elif stage_block_type == "moat":
                    block = MOATBlock(
                        hidden_size=self.hidden_size[stage_id],
                        expansion_rate=self.expansion_rate,
                        block_stride=block_stride,
                        pool_size=self.pool_size,
                        norm_class=self.norm_class,
                        activation=self.activation,
                        survival_prob=self.survival_prob,
                        head_size=self.head_size, #?
                        window_size=local_window_size,
                        relative_position_embedding_type=self.relative_position_embedding_type,
                        position_embedding_size=self.position_embedding_size[stage_id],
                        ln_epsilon=self.ln_epsilon,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        use_checkpointing_for_attention=self.use_checkpointing_for_attention,
                        name=block_name
                    )
                else:
                    raise ValueError(f"Unsupported block_type: {stage_block_type}")

                stage_blocks.append(block)

            self._blocks.append(stage_blocks)

        super().build(input_shape)

    def call(self, inputs, training=False, mask=None):
        
        output = self._stem(inputs, training=training)
        endpoints = [output]

        for _, stage_blocks in enumerate(self._blocks):
            for block in stage_blocks:
                output = block(output, training=training)
            endpoints += [output]

        if self.return_endpoints:
            assert len(endpoints) == 5

            return endpoints

        return output


def moat0(return_endpoints=False, use_pos_emb=True):
    return MOAT(
        stem_size=[64, 64],
        block_type_list=['mbconv', 'mbconv', 'moat', 'moat'],
        num_blocks=[2, 3, 7, 2],
        hidden_size=[96, 192, 384, 768],
        position_embedding_size=DEFAULT_POS_EMB_SIZE if use_pos_emb else None,
        survival_prob=0.8,
        return_endpoints=return_endpoints,
    )

def moat1(return_endpoints=False, use_pos_emb=True):
    return MOAT(
        stem_size=[64, 64],
        block_type_list=['mbconv', 'mbconv', 'moat', 'moat'],
        num_blocks=[2, 6, 14, 2],
        hidden_size=[96, 192, 384, 768],
        position_embedding_size=DEFAULT_POS_EMB_SIZE if use_pos_emb else None,
        survival_prob=0.7,
        return_endpoints=return_endpoints,
    )


def moat2(return_endpoints=False, use_pos_emb=True):
    return MOAT(
        stem_size=[128, 128],
        block_type_list=['mbconv', 'mbconv', 'moat', 'moat'],
        num_blocks=[2, 6, 14, 2],
        hidden_size=[128, 256, 512, 1024],
        position_embedding_size=DEFAULT_POS_EMB_SIZE if use_pos_emb else None,
        survival_prob=0.7,
        return_endpoints=return_endpoints,
    )


def moat3(return_endpoints=False, use_pos_emb=True):
    return MOAT(
        stem_size=[160, 160],
        block_type_list=['mbconv', 'mbconv', 'moat', 'moat'],
        num_blocks=[2, 12, 28, 2],
        hidden_size=[160, 320, 640, 1280],
        position_embedding_size=DEFAULT_POS_EMB_SIZE if use_pos_emb else None,
        survival_prob=0.4,
        return_endpoints=return_endpoints,
    )

def moat4(return_endpoints=False, use_pos_emb=True):
    return MOAT(
        stem_size=[256, 256],
        block_type_list=['mbconv', 'mbconv', 'moat', 'moat'],
        num_blocks=[2, 12, 28, 2],
        hidden_size=[256, 512, 1024, 2048],
        position_embedding_size=DEFAULT_POS_EMB_SIZE if use_pos_emb else None,
        survival_prob=0.3,
        return_endpoints=return_endpoints,
    )