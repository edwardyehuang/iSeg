import tensorflow as tf

from iseg.layers.model_builder import ImageLevelBlock, ConvBnRelu


class AtrousSpatialPyramidPooling(tf.keras.Model):
    def __init__(
        self, 
        filters=256, 
        receptive_fields=[6, 12, 18], 
        use_pixel_level=True, 
        use_image_level=True, 
        name=None
    ):

        super().__init__(name=name)

        self.filters = filters
        self.use_pixel_level = use_pixel_level
        self.use_image_level = use_image_level
        self.receptive_fields = receptive_fields

    
    def build(self, input_shape):
        
        if self.use_image_level:
            self.image_level_block = ImageLevelBlock(
                self.filters, name="image_level_block"
            )

        if self.use_pixel_level:
            self.pixel_level_block = ConvBnRelu(
                self.filters, (1, 1), name="pixel_level_block"
            )

        self.middle_convs = []

        for rate in self.receptive_fields:
            self.middle_convs.append(
                ConvBnRelu(
                    self.filters, 
                    (3, 3), 
                    dilation_rate=rate, 
                    name=f"rate_{rate}_conv"
                )
            )
        
        super().build(input_shape)


    def call(self, inputs, training=None):

        results = []

        if self.use_image_level:
            results.append(self.image_level_block(inputs, training=training))

        if self.use_pixel_level:
            results.append(self.pixel_level_block(inputs, training=training))

        for i in range(len(self.middle_convs)):
            results.append(self.middle_convs[i](inputs, training=training))

        results = tf.concat(results, axis=-1)

        return results