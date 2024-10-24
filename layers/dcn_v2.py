# This code is motified from https://github.com/RuaHU/keras_DCNv2
# Now support full XLA JIT compilation


import tensorflow as tf
import keras

from iseg.utils import get_tensor_shape

from iseg.utils.keras3_utils import Keras3_Layer_Wrapper

class DCNv2(Keras3_Layer_Wrapper):
    def __init__(
        self, filters, 
        kernel_size, 
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        use_custom_offset=False,
        activation=None,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = (1, 1, 1, 1)
        self.dilation = (1, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.use_custom_offset = use_custom_offset

        self.activation = keras.activations.get(activation)


    def build(self, input_shape):
        
        if self.use_custom_offset:
            input_channels = input_shape[0][-1]
        else:
            input_channels = input_shape[-1]

        self.kernel = self.add_weight(
            name='kernel',
            shape=self.kernel_size + (input_channels, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype='float32',
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
            )
        
        #[kh, kw, ic, 3 * groups * kh, kw]--->3 * groups * kh * kw = oc [output channels]
        self.offset_kernel = self.add_weight(
            name='offset_kernel',
            shape=self.kernel_size + (input_channels, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]), 
            initializer='zeros',
            trainable=True,
            dtype='float32'
        )
        
        self.offset_bias = self.add_weight(
            name='offset_bias',
            shape=(3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer='zeros',
            trainable=True,
            dtype='float32',
        )

        self.ks = self.kernel_size[0] * self.kernel_size[1]
        self.ph, self.pw = (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype = 'int32')
        self.patch_yx = tf.stack(tf.meshgrid(tf.range(-self.phw[1], self.phw[1] + 1), tf.range(-self.phw[0], self.phw[0] + 1))[::-1], axis = -1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])

        super().build(input_shape)


    # @tf.function(jit_compile=True, autograph=False)
    def _forward(self, x, offset):

        #x: [B, H, W, C]
        #offset: [B, H, W, ic] convx [kh, kw, ic, 3 * groups * kh * kw] ---> [B, H, W, 3 * groups * kh * kw]
        offset = tf.nn.conv2d(offset, self.offset_kernel, strides=self.stride, padding="SAME")
        offset = tf.add(offset, self.offset_bias, name="offset.add.bias")
        bs, ih, iw, ic = get_tensor_shape(x)

        #[B, H, W, 18], [B, H, W, 9]
        oyox, mask = offset[..., :2 * self.ks], offset[..., 2 * self.ks:]

        oyox = tf.reshape(oyox, [bs, ih, iw, self.ks, 2])

        mask = tf.nn.sigmoid(mask)
        #[H, W, 2]
        grid_yx = tf.stack(tf.meshgrid(tf.range(iw), tf.range(ih))[::-1], axis=-1)
        #[1, H, W, 9, 2]
        grid_yx = tf.reshape(grid_yx, [1, ih, iw, 1, 2]) + self.phw + self.patch_yx
        #[B, H, W, 9, 2]
        grid_yx = tf.add(tf.cast(grid_yx, oyox.dtype), oyox, name="grid.add.oyox")

        grid_iy0ix0 = tf.floor(grid_yx)
        grid_iy1ix1 = tf.clip_by_value(grid_iy0ix0 + tf.ones_like(grid_iy0ix0), 0, tf.constant([ih+1, iw+1], dtype=grid_iy0ix0.dtype)) # 

        #[B, H, W, 9, 1] * 2
        grid_iy1, grid_ix1 = tf.split(grid_iy1ix1, 2, axis=4)
        grid_iy0ix0 = tf.clip_by_value(grid_iy0ix0, 0, tf.constant([ih+1, iw+1], dtype=grid_iy0ix0.dtype))
        grid_iy0, grid_ix0 = tf.split(grid_iy0ix0, 2, axis=4)

        grid_yx = tf.clip_by_value(grid_yx, 0, tf.constant([ih+1, iw+1], dtype=grid_yx.dtype))

        #[B, H, W, 9, 4, 1]
        batch_index = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1, 1, 1, 1]), [1, ih, iw, self.ks, 4, 1])

        
        grid = tf.concat(
            [grid_iy1ix1, grid_iy1, grid_ix0, grid_iy0, grid_ix1, grid_iy0ix0], 
            axis=-1,
            name="all.grid.concat0",
        ) # [B, H, W, 9, 8]
        
        #[B, H, W, 9, 4, 2]
        grid = tf.reshape(grid, [bs, ih, iw, self.ks, 4, 2])
        #[B, H, W, 9, 4, 3]

        grid = tf.concat([batch_index, tf.cast(grid, tf.int32)], axis=-1, name="grid.concat.batch.index")

        diff_grid_0 = grid_yx - grid_iy0ix0 # [B, H, W, 9, 2]
        diff_grid_1 = grid_iy1ix1 - grid_yx # [B, H, W, 9, 2]

        diff_grid_0_0, diff_grid_0_1 = tf.unstack(diff_grid_0, 2, axis=-1) # [B, H, W, 9]
        diff_grid_1_0, diff_grid_1_1 = tf.unstack(diff_grid_1, 2, axis=-1) # [B, H, W, 9]

        diff_grid_concat = tf.stack([diff_grid_0_0, diff_grid_0_1, diff_grid_1_0, diff_grid_1_1], axis=-1, name="diff.grid.stack") # [B, H, W, 9, 4]

        #[B, H, W, 9, 2, 2]
        delta = tf.reshape(diff_grid_concat, [bs, ih, iw, self.ks, 2, 2]) # [B, H, W, 9, 2, 2]

        w0, w1 = tf.unstack(delta, 2, axis=-1) # [B, H, W, 9, 2], [B, H, W, 9, 2]
        w0 = tf.expand_dims(w0, axis=-1) # [B, H, W, 9, 2, 1]
        w1 = tf.expand_dims(w1, axis=-2) # [B, H, W, 9, 1, 2]

        w = w0 * w1 # [B, H, W, 9, 2, 2]

        #[B, H+2, W+2, C]
        x = tf.pad(x, [[0, 0], [int(self.ph), int(self.ph)], [int(self.pw), int(self.pw)], [0, 0]])
        
        mask = tf.expand_dims(mask, axis=-1) # [B, H, W, 9, 1]
        mask = tf.unstack(mask, self.ks, axis=3) # [B, H, W, 1] * 9

        map_sample = tf.gather_nd(x, grid) # [B, H, W, 9, 4, C]
        map_sample = tf.unstack(map_sample, self.ks, axis=3) # [B, H, W, 4, C] * 9

        w = tf.reshape(w, [bs, ih, iw, self.ks, 4]) # [B, H, W, 9, 4]
        w = tf.unstack(w, self.ks, axis=3) # [B, H, W, 4] * 9
                       
        map_bilinear = [None] * self.ks

        for i in range(self.ks):
            _w = w[i] # [B, H, W, 4]
            _w = tf.expand_dims(_w, axis=-2) # [B, H, W, 1, 4]
            _map_sample = map_sample[i] # [B, H, W, 4, C]
            _mask = mask[i] # [B, H, W, 1]

            _map_bilinear = tf.matmul(_w, _map_sample) # [B, H, W, 1, C]
            _map_bilinear = tf.squeeze(_map_bilinear, axis=-2)

            _map_bilinear = tf.multiply(_map_bilinear, _mask) # [B, H, W, C]

            map_bilinear[i] = _map_bilinear

        map_bilinear = tf.stack(map_bilinear, axis=-2) # [B, H, W, 9, C]

        #[B, H, W, 9*C]
        map_all = tf.reshape(map_bilinear, [bs, ih, iw, -1])
        #[B, H, W, OC]
        output = tf.nn.conv2d(map_all, tf.reshape(self.kernel, [1, 1, -1, self.filters]), strides=self.stride, padding='SAME')

        if self.use_bias:
            output += tf.cast(self.bias, output.dtype)

        output = self.activation(output)

        return output
        
        
    def call(self, inputs, training=None):

        if self.use_custom_offset:
            x, offset = tuple(inputs)
        else:
            x = offset = inputs

        return self._forward(x, offset)
        
        
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)