import tensorflow as tf 
import keras

from iseg.utils.version_utils import is_keras3

if not is_keras3():
    from keras.src.utils.losses_utils import ReductionV2 # type: ignore

from iseg.utils.common import get_tensor_shape
from iseg.utils.tensor_utils import get_stable_float_dtype_for_loss

class SegLossBase (keras.losses.Loss):

    def __init__(
        self,
        num_class=21,
        ignore_label=255,
        batch_size=2,
        reduction=False,
        from_logits=True,
        class_weights=None,
        name=None,
    ):
        
        if isinstance(reduction, bool):
            if is_keras3():
                reduction = "sum_over_batch_size" if not reduction else None
            else:
                reduction = ReductionV2.AUTO if not reduction else ReductionV2.NONE

        super().__init__(
            name=name,
            reduction=reduction,
        )

        self.num_class = num_class
        self.ignore_label = ignore_label
        self.batch_size = batch_size
        self.from_logits = from_logits
        self.class_weights = class_weights


    def call (self, y_true, y_pred):

        return self.internal_call(y_true, y_pred)


    @tf.autograph.experimental.do_not_convert
    def internal_call (self, y_true, y_pred):

        float_dtype = get_stable_float_dtype_for_loss()

        y_true = tf.cast(y_true, tf.int32) # [batch, h, w]
        y_pred = tf.cast(y_pred, float_dtype) # [batch, h, w, num_class]

        batch_size, height, width, _ = get_tensor_shape(y_pred)

        y_true = tf.cast(tf.expand_dims(y_true, axis=-1), float_dtype) # [batch, h, w, 1]
        y_true = tf.image.resize(y_true, [height, width], method="nearest") # [batch, h, w, 1]
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32) # [batch, h, w]

        valid_mask = self.compute_valid_mask(y_true)
        y_true, y_pred = self.before_compute_loss_forward(y_true, y_pred)

        valid_mask = tf.cast(valid_mask, float_dtype)
        valid_mask = tf.reshape(valid_mask, [batch_size, -1]) # [batch, h * w]

        return self.compute_loss_forwards(y_true, y_pred, valid_mask=valid_mask)
    

    def compute_valid_mask (self, y_true):
        
        return tf.math.not_equal(y_true, self.ignore_label) # [batch, h, w]
    

    def before_compute_loss_forward (self, y_true, y_pred):

        return y_true, y_pred
    

    def compute_loss_forwards (self, y_true, y_pred, valid_mask=None):

        raise NotImplementedError("compute_loss_forwards() is not implemented.")