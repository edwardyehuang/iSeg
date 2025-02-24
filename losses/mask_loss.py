import tensorflow as tf 
import keras

from iseg.utils.version_utils import is_keras3

if not is_keras3():
    from keras.src.utils.losses_utils import ReductionV2

from iseg.utils.common import get_tensor_shape
from iseg.utils.value_check import check_numerics

class MaskLoss (keras.losses.Loss):
    
    def __init__(
        self,
        num_class=21,
        ignore_label=255,
        batch_size=2,
        reduction=False,
        from_logits=True,
        class_weights=None,
        use_sigmoid_loss=True,
        use_dice_loss=True,
        use_ce_loss=True,
        ce_loss_coefficient=1.0,
        sigmoid_loss_coefficient=20.0,
        dice_loss_coefficient=1.0,
        apply_focal_sigmoid_loss=True,
        apply_focal_ce_loss=False,
        apply_class_balancing=False,
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

        self.use_sigmoid_loss = use_sigmoid_loss
        self.use_dice_loss = use_dice_loss
        self.use_ce_loss = use_ce_loss

        self.ce_loss_coefficient = ce_loss_coefficient
        self.sigmoid_loss_coefficient = sigmoid_loss_coefficient
        self.dice_loss_coefficient = dice_loss_coefficient

        self.apply_focal_sigmoid_loss = apply_focal_sigmoid_loss
        self.apply_focal_ce_loss = apply_focal_ce_loss

        self.apply_class_balancing = apply_class_balancing

        print("MaskLoss: use_sigmoid_loss = ", self.use_sigmoid_loss)
        print("MaskLoss: use_dice_loss = ", self.use_dice_loss)
        print("MaskLoss: use_ce_loss = ", self.use_ce_loss)
        print("MaskLoss: ce_loss_coefficient = ", self.ce_loss_coefficient)
        print("MaskLoss: sigmoid_loss_coefficient = ", self.sigmoid_loss_coefficient)
        print("MaskLoss: dice_loss_coefficient = ", self.dice_loss_coefficient)
        print("MaskLoss: apply_focal_sigmoid_loss = ", self.apply_focal_sigmoid_loss)
        print("MaskLoss: apply_focal_ce_loss = ", self.apply_focal_ce_loss)
        print("MaskLoss: apply_class_balancing = ", self.apply_class_balancing)


    def call (self, y_true, y_pred):

        return self.internal_call(y_true, y_pred)
    

    @tf.autograph.experimental.do_not_convert
    def internal_call (self, y_true, y_pred):

        y_true = tf.cast(y_true, tf.int32) # [batch, h, w]
        y_pred = tf.cast(y_pred, tf.float32) # [batch, h, w, num_class]

        batch_size, height, width, _ = get_tensor_shape(y_pred)

        y_true = tf.cast(tf.expand_dims(y_true, axis=-1), tf.float32) # [batch, h, w, 1]
        y_true = tf.image.resize(y_true, [height, width], method="nearest") # [batch, h, w, 1]
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32) # [batch, h, w]

        valid_mask = self.compute_valid_mask(y_true)
        y_true, y_pred = self.before_compute_loss_forward(y_true, y_pred)

        valid_mask = tf.cast(valid_mask, tf.float32)
        valid_mask = tf.reshape(valid_mask, [batch_size, -1]) # [batch, h * w]

        return self.compute_loss_forwards(y_true, y_pred, valid_mask=valid_mask)
    

    def compute_valid_mask (self, y_true):
        
        return tf.math.not_equal(y_true, self.ignore_label) # [batch, h, w]
    

    def before_compute_loss_forward (self, y_true, y_pred):

        return y_true, y_pred


    @tf.autograph.experimental.do_not_convert
    def compute_loss_forwards (self, y_true, y_pred, valid_mask=None):

        # y_true [batch, h, w]
        # y_pred [batch, h, w, num_class]

        batch_size, height, width, num_class = get_tensor_shape(y_pred)

        if self.ignore_label == 0:
            y_true -= 1

        y_true_one_hot = tf.one_hot(y_true, self.num_class) # [batch, h, w, num_class]
        y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)
        y_true_one_hot = tf.transpose(y_true_one_hot, [0, 3, 1, 2]) # [batch, num_class, h, w]
        y_true_one_hot = tf.reshape(y_true_one_hot, [batch_size, num_class, -1]) # [batch, num_class, h * w]

        y_pred = tf.transpose(y_pred, [0, 3, 1, 2]) # [batch, num_class, h, w]
        y_pred = tf.reshape(y_pred, [batch_size, num_class, -1]) # [batch, num_class, h * w]

        loss_list = []

        # compute sigmoid loss

        if self.use_sigmoid_loss:
            if self.apply_focal_sigmoid_loss:
                sigmoid_loss = keras.losses.binary_focal_crossentropy(
                    y_true_one_hot, 
                    y_pred,
                    from_logits=self.from_logits, 
                    apply_class_balancing=self.apply_class_balancing,
                    axis=1 # reduce over num_class [batch, h * w]
                )
            else:
                sigmoid_loss = keras.losses.binary_crossentropy(
                    y_true_one_hot, 
                    y_pred, 
                    from_logits=self.from_logits, 
                    apply_class_balancing=self.apply_class_balancing,
                    axis=1
                )

            sigmoid_loss = check_numerics(sigmoid_loss, message="sigmoid_loss contains nan or inf")

            sigmoid_loss *= self.sigmoid_loss_coefficient
            loss_list.append(sigmoid_loss)

        # compute dice loss

        if self.use_dice_loss:
            dice_loss = dice(y_true_one_hot, y_pred, from_logits=self.from_logits, weighted_mask=valid_mask) # [batch]
            dice_loss = tf.expand_dims(dice_loss, axis=-1) # [batch, 1]
            dice_loss *= self.dice_loss_coefficient
            dice_loss += tf.zeros([batch_size, height * width], dtype=dice_loss.dtype) # [batch, h * w]
            dice_loss = check_numerics(dice_loss, message="dice_loss contains nan or inf")
            loss_list.append(dice_loss)

        # compute ce loss

        if self.use_ce_loss:
            y_true_one_hot = tf.transpose(y_true_one_hot, [0, 2, 1]) # [batch, h * w, num_class]
            y_true_one_hot = tf.reshape(y_true_one_hot, [-1, num_class]) # [batch * h * w, num_class]

            y_pred = tf.transpose(y_pred, [0, 2, 1]) # [batch, h * w, num_class]
            y_pred = tf.reshape(y_pred, [-1, num_class]) # [batch * h * w, num_class]

            if self.apply_focal_ce_loss:
                ce_loss = keras.losses.categorical_focal_crossentropy(
                    y_true_one_hot, y_pred, from_logits=self.from_logits
                ) # [batch * h * w]
            else:
                ce_loss = keras.losses.categorical_crossentropy(
                    y_true_one_hot, y_pred, from_logits=self.from_logits
                )

            ce_loss = tf.reshape(ce_loss, [batch_size, -1]) # [batch, h * w]
            ce_loss = check_numerics(ce_loss, message="ce_loss contains nan or inf")
            ce_loss *= self.ce_loss_coefficient
            loss_list.append(ce_loss)

        loss = tf.add_n(loss_list)
        loss._keras_mask = valid_mask

        return loss
    

def dice(y_true, y_pred, from_logits=False, weighted_mask=None):
    """Computes the Dice loss value between `y_true` and `y_pred`.

    Formula:
    ```python
    loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
    ```

    Args:
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.

    Returns:
        Dice loss value.
    """
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    batch_size = tf.shape(y_pred)[0]

    if from_logits:
        y_pred = tf.math.sigmoid(y_pred)

    if weighted_mask is not None:
        weighted_mask = tf.cast(weighted_mask, y_pred.dtype)
        weighted_mask = tf.expand_dims(weighted_mask, axis=1) # [batch, 1, h * w]
        y_pred *= weighted_mask
        y_true *= weighted_mask

    inputs = tf.reshape(y_true, [batch_size, -1])
    targets = tf.reshape(y_pred, [batch_size, -1])

    intersection = tf.reduce_sum(inputs * targets, axis=-1)
    intersection *= 2.0
    intersection += keras.backend.epsilon()

    den = tf.reduce_sum(targets, axis=-1) + tf.reduce_sum(inputs, axis=-1) + keras.backend.epsilon()

    dice = tf.math.divide_no_nan(
        intersection, den,
    )

    return 1 - dice
        
