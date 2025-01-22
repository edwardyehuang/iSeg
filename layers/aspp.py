import tensorflow as tf

from iseg.layers.model_builder import ImageLevelBlock, ConvNormAct
from iseg.utils.keras3_utils import Keras3_Model_Wrapper


class AtrousSpatialPyramidPooling(Keras3_Model_Wrapper):
    def __init__(
        self, 
        filters=256, 
        dilation_rates=[3, 6, 9], 
        dilation_rates_multiplier=1,
        use_pixel_level=True, 
        use_image_level=True, 
        name=None
    ):

        super().__init__(name=name)

        self.filters = filters
        self.use_pixel_level = use_pixel_level
        self.use_image_level = use_image_level
        self.dilation_rates = dilation_rates
        self.dilation_rates_multiplier = dilation_rates_multiplier

    
    def build(self, input_shape):
        
        if self.use_image_level:
            self.image_level_block = ImageLevelBlock(
                self.filters, name="image_level_block"
            )

        if self.use_pixel_level:
            self.pixel_level_block = ConvNormAct(
                self.filters, (1, 1), name="pixel_level_block"
            )

        self.asp_convs = []

        for rate in self.dilation_rates:
            
            rate = rate * self.dilation_rates_multiplier

            self.asp_convs.append(
                ConvNormAct(
                    self.filters, 
                    (3, 3), 
                    dilation_rate=rate, 
                    name=f"asp_convs_{rate}"
                )
            )
        
        super().build(input_shape)


    def call(self, inputs, training=None):

        results = []

        if self.use_image_level:
            results.append(self.image_level_block(inputs, training=training))

        if self.use_pixel_level:
            results.append(self.pixel_level_block(inputs, training=training))

        for i in range(len(self.asp_convs)):
            results.append(self.asp_convs[i](inputs, training=training))

        results = tf.concat(results, axis=-1)

        return results