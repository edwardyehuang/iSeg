import tensorflow as tf
import os
import numpy as np

# Load the custom op library
_dcn_v2_module = None

def _load_dcn_v2_op():
    global _dcn_v2_module
    if _dcn_v2_module is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(current_dir, 'dcn_v2_op.so')
        if os.path.exists(lib_path):
            try:
                _dcn_v2_module = tf.load_op_library(lib_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load DCNv2 op library: {e}")
        else:
            raise RuntimeError(f"DCNv2 op library not found at {lib_path}. Please compile the op first.")
    return _dcn_v2_module

# Load the module when this file is imported
try:
    # Set up library path for TensorFlow framework
    import subprocess
    import sys
    
    # Get TensorFlow library path
    tf_lib_path = tf.sysconfig.get_lib()
    if tf_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
        os.environ['LD_LIBRARY_PATH'] = f"{tf_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    _dcn_v2_module = _load_dcn_v2_op()
    print("DCNv2 TensorFlow op library loaded successfully!")
except Exception as e:
    print(f"Warning: Failed to load DCNv2 op library: {e}")
    _dcn_v2_module = None

def dcn_v2_conv2d(input_tensor, offset, mask, weight, bias=None,
                  kernel_size=(3, 3), padding='SAME', strides=(1, 1),
                  dilation_rate=(1, 1), deformable_groups=1):
    """
    Deformable Convolution v2 using custom CUDA op.
    
    Args:
        input_tensor: Input tensor of shape [batch, height, width, channels]
        offset: Offset tensor of shape [batch, height, width, 2*kernel_h*kernel_w*deformable_groups]
        mask: Mask tensor of shape [batch, height, width, kernel_h*kernel_w*deformable_groups]
        weight: Weight tensor of shape [kernel_h, kernel_w, input_channels, output_channels]
        bias: Bias tensor of shape [output_channels] (optional)
        kernel_size: Tuple of (kernel_h, kernel_w)
        padding: Either 'SAME' or 'VALID'
        strides: Tuple of (stride_h, stride_w)
        dilation_rate: Tuple of (dilation_h, dilation_w)
        deformable_groups: Number of deformable groups
    
    Returns:
        Output tensor of shape [batch, output_height, output_width, output_channels]
    """
    dcn_v2_module = _dcn_v2_module
    
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilation_rate
    
    # Calculate padding
    if padding == 'SAME':
        input_height = tf.shape(input_tensor)[1]
        input_width = tf.shape(input_tensor)[2]
        
        if stride_h == 1:
            pad_h = max(0, (kernel_h - 1) * dilation_h) // 2
        else:
            pad_h = max(0, (kernel_h - 1) * dilation_h) // 2
            
        if stride_w == 1:
            pad_w = max(0, (kernel_w - 1) * dilation_w) // 2
        else:
            pad_w = max(0, (kernel_w - 1) * dilation_w) // 2
    else:  # VALID
        pad_h = pad_w = 0
    
    if bias is None:
        bias = tf.zeros([tf.shape(weight)[-1]], dtype=input_tensor.dtype)
    
    # Use keyword arguments as required by the TensorFlow op
    output = dcn_v2_module.DCNv2Forward(
        input=input_tensor,
        offset=offset,
        mask=mask,
        weight=weight,
        bias=bias,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        pad_h=pad_h,
        pad_w=pad_w,
        stride_h=stride_h,
        stride_w=stride_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        deformable_groups=deformable_groups
    )
    
    return output


class DCNv2Optimized(tf.keras.layers.Layer):
    """
    Optimized Deformable Convolution v2 layer using custom CUDA ops.
    
    This implementation uses optimized CUDA kernels for better performance
    compared to the pure TensorFlow implementation.
    """
    
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 dilation_rate=(1, 1),
                 deformable_groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_custom_offset=False,
                 activation=None,
                 **kwargs):
        super(DCNv2Optimized, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides)
        self.padding = padding.upper()
        self.dilation_rate = dilation_rate if isinstance(dilation_rate, (list, tuple)) else (dilation_rate, dilation_rate)
        self.deformable_groups = deformable_groups
        self.use_bias = use_bias
        self.use_custom_offset = use_custom_offset
        
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        if self.use_custom_offset:
            input_channels = input_shape[0][-1]
        else:
            input_channels = input_shape[-1]
            
        # Main convolution kernel
        self.kernel = self.add_weight(
            name='kernel',
            shape=self.kernel_size + (input_channels, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype
            )
        else:
            self.bias = None
            
        # Offset and mask generation layers
        if not self.use_custom_offset:
            offset_mask_channels = 3 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
            
            self.offset_kernel = self.add_weight(
                name='offset_kernel',
                shape=self.kernel_size + (input_channels, offset_mask_channels),
                initializer='zeros',
                trainable=True,
                dtype=self.dtype
            )
            
            self.offset_bias = self.add_weight(
                name='offset_bias',
                shape=(offset_mask_channels,),
                initializer='zeros',
                trainable=True,
                dtype=self.dtype
            )
        
        super().build(input_shape)
    
    def _generate_offset_mask(self, x):
        """Generate offset and mask from input."""
        # Convolution to generate offset and mask
        offset_mask = tf.nn.conv2d(
            x, self.offset_kernel,
            strides=[1] + list(self.strides) + [1],
            padding=self.padding,
            dilations=[1] + list(self.dilation_rate) + [1]
        )
        offset_mask = tf.nn.bias_add(offset_mask, self.offset_bias)
        
        # Split into offset and mask
        ks = self.kernel_size[0] * self.kernel_size[1]
        offset_channels = 2 * self.deformable_groups * ks
        
        offset = offset_mask[..., :offset_channels]
        mask = offset_mask[..., offset_channels:]
        mask = tf.nn.sigmoid(mask)
        
        return offset, mask
    
    def call(self, inputs, training=None):
        if self.use_custom_offset:
            x, offset, mask = inputs
        else:
            x = inputs
            offset, mask = self._generate_offset_mask(x)
        
        # Use the optimized CUDA op
        output = dcn_v2_conv2d(
            input_tensor=x,
            offset=offset,
            mask=mask,
            weight=self.kernel,
            bias=self.bias,
            kernel_size=self.kernel_size,
            padding=self.padding,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            deformable_groups=self.deformable_groups
        )
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
    def compute_output_shape(self, input_shape):
        if self.use_custom_offset:
            batch_size = input_shape[0][0]
            input_height = input_shape[0][1]
            input_width = input_shape[0][2]
        else:
            batch_size = input_shape[0]
            input_height = input_shape[1]
            input_width = input_shape[2]
        
        if self.padding == 'SAME':
            output_height = (input_height + self.strides[0] - 1) // self.strides[0]
            output_width = (input_width + self.strides[1] - 1) // self.strides[1]
        else:  # VALID
            output_height = (input_height - (self.kernel_size[0] - 1) * self.dilation_rate[0]) // self.strides[0]
            output_width = (input_width - (self.kernel_size[1] - 1) * self.dilation_rate[1]) // self.strides[1]
        
        return tf.TensorShape([batch_size, output_height, output_width, self.filters])
    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding.lower(),
            'dilation_rate': self.dilation_rate,
            'deformable_groups': self.deformable_groups,
            'use_bias': self.use_bias,
            'use_custom_offset': self.use_custom_offset,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'activation': tf.keras.activations.serialize(self.activation),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
