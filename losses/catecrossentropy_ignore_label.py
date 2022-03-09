# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


def catecrossentropy_ignore_label_loss(
    num_class=21,
    ignore_label=255,
    class_weights=None,
    batch_size=2,
    reduction=False,
    pre_compute_fn=None,
    post_compute_fn=None,
    from_logits=True,
):

    loss_func = tf.keras.losses.CategoricalCrossentropy(
        from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE
    )

    def weighted_loss(y_true, y_pred):

        local_batch_size = tf.shape(y_pred)[0]

        if pre_compute_fn is not None:
            y_true, y_pred = pre_compute_fn(y_true, y_pred)

        y_true = tf.cast(y_true, tf.dtypes.int32)

        y_pred = tf.reshape(y_pred, shape=[-1, num_class]) # [NHW, class]
        y_true = tf.reshape(y_true, shape=[-1])

        not_ignore_mask = tf.math.not_equal(y_true, ignore_label) # [NHW]

        if ignore_label == 0:
            y_true -= 1

        one_hot_label = tf.one_hot(y_true, num_class) # [NHW, class]

        sample_weights = tf.identity(not_ignore_mask, name="sample_weights") # [NHW]
        

        if class_weights is not None and len(class_weights) > 0: # [class]
            
            assert len(class_weights) == num_class

            sample_weights = tf.cast(sample_weights, tf.float32)

            _class_weights = tf.expand_dims(class_weights, axis=0) # [1, class]
            _class_weights *= one_hot_label # [NHW, class]
            _class_weights = tf.reduce_sum(_class_weights, axis=-1) # [NHW]

            sample_weights *= tf.cast(_class_weights, tf.float32) # [NHW]


        loss_value = loss_func(one_hot_label, y_pred, sample_weights)

        if post_compute_fn is not None:
            loss_value = post_compute_fn(one_hot_label, y_pred, loss_value, local_batch_size)

        if reduction:
            loss_value = tf.nn.compute_average_loss(loss_value, global_batch_size=batch_size)

        return loss_value

    return weighted_loss
