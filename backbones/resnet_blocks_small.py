import tensorflow as tf

from iseg.layers.normalizations import normalization

BN_EPSILON = 1.001e-5

class BlockType2Small(tf.keras.Model):
    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        use_bias=False,
        norm_method=None,
        downsample_method="avg",
        name=None,
    ):

        super().__init__(name=name)

        self.filters = filters
        self.conv_shortcut = conv_shortcut
        self.downsample_method = downsample_method
        self.use_bias = use_bias
        self.norm_method = norm_method

        self.conv1_conv = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=stride, padding="SAME", use_bias=use_bias, name=name + "_1_conv"
        )
        self.conv1_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_1_bn")

        self.conv2_conv = tf.keras.layers.Conv2D(
            filters, kernel_size, padding="SAME", use_bias=use_bias, name=name + "_2_conv"
        )
        self.conv2_bn = normalization(epsilon=BN_EPSILON, method=norm_method, name=name + "_2_bn")


    def build (self, input_shape):

        if input_shape[-1] == self.filters:
            self.conv_shortcut = False

        if self.conv_shortcut:
            self.shortcut_conv = tf.keras.layers.Conv2D(
                self.filters, 
                kernel_size=1, 
                strides=self.conv1_conv.strides, 
                use_bias=self.use_bias, 
                name=self.name + "_0_conv"
            )
            self.shortcut_bn = normalization(epsilon=BN_EPSILON, method=self.norm_method, name=self.name + "_0_bn")

    @property
    def strides(self):
        return self.conv1_conv.strides[0]

    @strides.setter
    def strides(self, value):

        if not isinstance(value, tuple):
            value = (value, value)

        self.conv1_conv.strides = value

        if self.conv_shortcut:
            self.shortcut_conv.strides = value

    @property
    def atrous_rates(self):
        return self.conv1_conv.dilation_rate[0]

    @atrous_rates.setter
    def atrous_rates(self, value):

        if not isinstance(value, tuple):
            value = (value, value)

        self.conv1_conv.dilation_rate = value
        self.conv2_conv.dilation_rate = value

    def call(self, inputs, training=None, **kwargs):

        if self.conv_shortcut:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        elif self.strides > 1:
            if "avg" in self.downsample_method:
                shortcut = tf.nn.avg_pool2d(inputs, self.conv1_conv.strides, self.conv1_conv.strides, "SAME")
            elif "max" in self.downsample_method:
                shortcut = tf.nn.max_pool2d(inputs, self.conv1_conv.strides, self.conv1_conv.strides, "SAME")
            else:
                raise ValueError("Only max or avg are supported")
        else:
            shortcut = inputs

        x = self.conv1_conv(inputs)
        x = self.conv1_bn(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2_conv(x, training=training)
        x = self.conv2_bn(x, training=training)

        x = tf.add(shortcut, x)
        x = tf.nn.relu(x)

        return x
