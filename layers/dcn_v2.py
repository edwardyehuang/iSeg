# This code is motified from https://github.com/RuaHU/keras_DCNv2


import tensorflow as tf

from iseg.layers.model_builder import get_tensor_shape_v2

class DCNv2(tf.keras.layers.Layer):
    def __init__(
        self, filters, 
        kernel_size, 
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        use_custom_offset=False,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = (1, 1, 1, 1)
        self.dilation = (1, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer

        self.use_custom_offset = use_custom_offset

    def build(self, input_shape):
        
        if self.use_custom_offset:
            input_channels = input_shape[0][-1]
        else:
            input_channels = input_shape[-1]

        self.kernel = self.add_weight(
            name = 'kernel',
            shape = self.kernel_size + (input_channels, self.filters),
            initializer = self.kernel_initializer,
            regularizer = self.kernel_regularizer,
            trainable = True,
            dtype = 'float32',
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name = 'bias',
                shape = (self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
            )
        
        #[kh, kw, ic, 3 * groups * kh, kw]--->3 * groups * kh * kw = oc [output channels]
        self.offset_kernel = self.add_weight(
            name = 'offset_kernel',
            shape = self.kernel_size + (input_channels, 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]), 
            initializer = 'zeros',
            trainable = True,
            dtype = 'float32'
        )
        
        self.offset_bias = self.add_weight(
            name = 'offset_bias',
            shape = (3 * self.kernel_size[0] * self.kernel_size[1] * self.deformable_groups,),
            initializer='zeros',
            trainable = True,
            dtype = 'float32',
        )

        self.ks = self.kernel_size[0] * self.kernel_size[1]
        self.ph, self.pw = (self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype = 'int32')
        self.patch_yx = tf.stack(tf.meshgrid(tf.range(-self.phw[1], self.phw[1] + 1), tf.range(-self.phw[0], self.phw[0] + 1))[::-1], axis = -1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])

        super().build(input_shape)
        
        
    def call(self, inputs, training=None):

        if self.use_custom_offset:
            x, offset = tuple(inputs)
        else:
            x = offset = inputs
        
        #x: [B, H, W, C]
        #offset: [B, H, W, ic] convx [kh, kw, ic, 3 * groups * kh * kw] ---> [B, H, W, 3 * groups * kh * kw]
        offset = tf.nn.conv2d(offset, self.offset_kernel, strides = self.stride, padding = 'SAME')
        offset += self.offset_bias
        bs, ih, iw, ic = get_tensor_shape_v2(x)

        #[B, H, W, 18], [B, H, W, 9]
        oyox, mask = offset[..., :2*self.ks], offset[..., 2*self.ks:]
        mask = tf.nn.sigmoid(mask)
        #[H, W, 2]
        grid_yx = tf.stack(tf.meshgrid(tf.range(iw), tf.range(ih))[::-1], axis = -1)
        #[1, H, W, 9, 2]
        grid_yx = tf.reshape(grid_yx, [1, ih, iw, 1, 2]) + self.phw + self.patch_yx
        #[B, H, W, 9, 2]
        grid_yx = tf.cast(grid_yx, 'float32') + tf.reshape(oyox, [bs, ih, iw, -1, 2])
        grid_iy0ix0 = tf.floor(grid_yx)
        grid_iy1ix1 = tf.clip_by_value(grid_iy0ix0 + 1, 0, tf.constant([ih+1, iw+1], dtype = 'float32'))
        #[B, H, W, 9, 1] * 2
        grid_iy1, grid_ix1 = tf.split(grid_iy1ix1, 2, axis = 4)
        grid_iy0ix0 = tf.clip_by_value(grid_iy0ix0, 0, tf.constant([ih+1, iw+1], dtype = 'float32'))
        grid_iy0, grid_ix0 = tf.split(grid_iy0ix0, 2, axis = 4)
        grid_yx = tf.clip_by_value(grid_yx, 0, tf.constant([ih+1, iw+1], dtype = 'float32'))
        #[B, H, W, 9, 4, 1]
        batch_index = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1, 1, 1, 1]), [1, ih, iw, self.ks, 4, 1])
        #[B, H, W, 9, 4, 2]
        grid = tf.reshape(tf.concat([grid_iy1ix1, grid_iy1, grid_ix0, grid_iy0, grid_ix1, grid_iy0ix0], axis = -1), [bs, ih, iw, self.ks, 4, 2])
        #[B, H, W, 9, 4, 3]
        grid = tf.concat([batch_index, tf.cast(grid, 'int32')], axis = -1)
        #[B, H, W, 9, 2, 2]
        delta = tf.reshape(tf.concat([grid_yx - grid_iy0ix0, grid_iy1ix1 - grid_yx], axis = -1), [bs, ih, iw, self.ks, 2, 2])
        #[B, H, W, 9, 2, 1] * [B, H, W, 9, 1, 2] = [B, H, W, 9, 2, 2]
        w = tf.expand_dims(delta[..., 0], axis = -1) * tf.expand_dims(delta[..., 1], axis = -2)
        #[B, H+2, W+2, C]
        x = tf.pad(x, [[0, 0], [int(self.ph), int(self.ph)], [int(self.pw), int(self.pw)], [0, 0]])
        #[B, H, W, 9, 4, C]
        map_sample = tf.gather_nd(x, grid)
        #([B, H, W, 9, 4, 1] * [B, H, W, 9, 4, C]).SUM(-2) * [B, H, W, 9, 1] = [B, H, W, 9, C]
        map_bilinear = tf.reduce_sum(tf.reshape(w, [bs, ih, iw, self.ks, 4, 1]) * map_sample, axis = -2) * tf.expand_dims(mask, axis = -1)
        #[B, H, W, 9*C]
        map_all = tf.reshape(map_bilinear, [bs, ih, iw, -1])
        #[B, H, W, OC]
        output = tf.nn.conv2d(map_all, tf.reshape(self.kernel, [1, 1, -1, self.filters]), strides = self.stride, padding = 'SAME')
        if self.use_bias:
            output += self.bias
        return output
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)