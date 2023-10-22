# ===================================================================
# MIT License
# Copyright (c) 2023 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================


from distutils.version import LooseVersion

import tensorflow as tf

import warnings

if LooseVersion(tf.version.VERSION) < LooseVersion("2.14.0"):
    from keras.backend import _get_logits
    from keras.losses import LossFunctionWrapper
else:
    from keras.src.backend import _get_logits
    from keras.src.losses import LossFunctionWrapper


def _categorical_focal_crossentropy(
    target,
    output,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    axis=-1,
):
    """Computes the alpha balanced focal crossentropy loss.

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. The general formula for the focal loss (FL)
    is as follows:

    `FL(p_t) = (1 − p_t)^gamma * log(p_t)`

    where `p_t` is defined as follows:
    `p_t = output if y_true == 1, else 1 - output`

    `(1 − p_t)^gamma` is the `modulating_factor`, where `gamma` is a focusing
    parameter. When `gamma` = 0, there is no focal effect on the cross entropy.
    `gamma` reduces the importance given to simple examples in a smooth manner.

    The authors use alpha-balanced variant of focal loss (FL) in the paper:
    `FL(p_t) = −alpha * (1 − p_t)^gamma * log(p_t)`

    where `alpha` is the weight factor for the classes. If `alpha` = 1, the
    loss won't be able to handle class imbalance properly as all
    classes will have the same weight. This can be a constant or a list of
    constants. If alpha is a list, it must have the same length as the number
    of classes.

    The formula above can be generalized to:
    `FL(p_t) = alpha * (1 − p_t)^gamma * CrossEntropy(target, output)`

    where minus comes from `CrossEntropy(target, output)` (CE).

    Extending this to multi-class case is straightforward:
    `FL(p_t) = alpha * (1 − p_t)^gamma * CategoricalCE(target, output)`

    Args:
        target: Ground truth values from the dataset.
        output: Predictions of the model.
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple examples in a smooth manner.
        from_logits: Whether `output` is expected to be a logits tensor. By
            default, we consider that `output` encodes a probability
            distribution.
        axis: Int specifying the channels axis. `axis=-1` corresponds to data
             format `channels_last`, and `axis=1` corresponds to data format
             `channels_first`.

    Returns:
        A tensor.
    """
    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    target.shape.assert_is_compatible_with(output.shape)

    output, from_logits =  _get_logits(
        output, from_logits, "Softmax", "categorical_focal_crossentropy"
    )

    if from_logits:
        output = tf.nn.softmax(output, axis=axis)

    # Adjust the predictions so that the probability of
    # each class for every sample adds up to 1
    # This is needed to ensure that the cross entropy is
    # computed correctly.
    output = output / tf.reduce_sum(output, axis=axis, keepdims=True)

    epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Calculate cross entropy
    cce = -target * tf.math.log(output)

    # Calculate factors
    modulating_factor = tf.pow(1.0 - output, gamma)
    weighting_factor = tf.multiply(modulating_factor, alpha)

    # Apply weighting factor
    focal_cce = tf.multiply(weighting_factor, cce)
    focal_cce = tf.reduce_sum(focal_cce, axis=axis)
    return focal_cce



def categorical_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
):
    """Computes the categorical focal crossentropy loss.

    Standalone usage:
    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.9, 0.05], [0.1, 0.85, 0.05]]
    >>> loss = tf.keras.losses.categorical_focal_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([2.63401289e-04, 6.75912094e-01], dtype=float32)

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple examples in a smooth manner. When `gamma` = 0, there is
            no focal effect on the categorical crossentropy.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By
            default, we assume that `y_pred` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels. For
            example, if `0.1`, use `0.1 / num_classes` for non-target labels
            and `0.9 + 0.1 / num_classes` for target labels.
        axis: Defaults to -1. The dimension along which the entropy is
            computed.

    Returns:
        Categorical focal crossentropy loss value.
    """
    if isinstance(axis, bool):
        raise ValueError(
            "`axis` must be of type `int`. "
            f"Received: axis={axis} of type {type(axis)}"
        )
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)

    if y_pred.shape[-1] == 1:
        warnings.warn(
            "In loss categorical_focal_crossentropy, expected "
            "y_pred.shape to be (batch_size, num_classes) "
            f"with num_classes > 1. Received: y_pred.shape={y_pred.shape}. "
            "Consider using 'binary_crossentropy' if you only have 2 classes.",
            SyntaxWarning,
            stacklevel=2,
        )

    def _smooth_labels():
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (
            label_smoothing / num_classes
        )

    y_true = tf.__internal__.smart_cond.smart_cond(
        label_smoothing, _smooth_labels, lambda: y_true
    )

    return _categorical_focal_crossentropy(
        target=y_true,
        output=y_pred,
        alpha=alpha,
        gamma=gamma,
        from_logits=from_logits,
        axis=axis,
    )



class CategoricalFocalCrossentropy(LossFunctionWrapper):
    """Computes the alpha balanced focal crossentropy loss.

    Use this crossentropy loss function when there are two or more label
    classes and if you want to handle class imbalance without using
    `class_weights`. We expect labels to be provided in a `one_hot`
    representation.

    According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
    helps to apply a focal factor to down-weight easy examples and focus more on
    hard examples. The general formula for the focal loss (FL)
    is as follows:

    `FL(p_t) = (1 − p_t)^gamma * log(p_t)`

    where `p_t` is defined as follows:
    `p_t = output if y_true == 1, else 1 - output`

    `(1 − p_t)^gamma` is the `modulating_factor`, where `gamma` is a focusing
    parameter. When `gamma` = 0, there is no focal effect on the cross entropy.
    `gamma` reduces the importance given to simple examples in a smooth manner.

    The authors use alpha-balanced variant of focal loss (FL) in the paper:
    `FL(p_t) = −alpha * (1 − p_t)^gamma * log(p_t)`

    where `alpha` is the weight factor for the classes. If `alpha` = 1, the
    loss won't be able to handle class imbalance properly as all
    classes will have the same weight. This can be a constant or a list of
    constants. If alpha is a list, it must have the same length as the number
    of classes.

    The formula above can be generalized to:
    `FL(p_t) = alpha * (1 − p_t)^gamma * CrossEntropy(y_true, y_pred)`

    where minus comes from `CrossEntropy(y_true, y_pred)` (CE).

    Extending this to multi-class case is straightforward:
    `FL(p_t) = alpha * (1 − p_t)^gamma * CategoricalCE(y_true, y_pred)`

    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.

    Standalone usage:

    >>> y_true = [[0., 1., 0.], [0., 0., 1.]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = tf.keras.losses.CategoricalFocalCrossentropy()
    >>> cce(y_true, y_pred).numpy()
    0.23315276

    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
    0.1632

    >>> # Using 'sum' reduction type.
    >>> cce = tf.keras.losses.CategoricalFocalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> cce(y_true, y_pred).numpy()
    0.46631

    >>> # Using 'none' reduction type.
    >>> cce = tf.keras.losses.CategoricalFocalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> cce(y_true, y_pred).numpy()
    array([3.2058331e-05, 4.6627346e-01], dtype=float32)

    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalFocalCrossentropy())
    ```

    Args:
        alpha: A weight balancing factor for all classes, default is `0.25` as
            mentioned in the reference. It can be a list of floats or a scalar.
            In the multi-class case, alpha may be set by inverse class
            frequency by using `compute_class_weight` from `sklearn.utils`.
        gamma: A focusing parameter, default is `2.0` as mentioned in the
            reference. It helps to gradually reduce the importance given to
            simple (easy) examples in a smooth manner.
        from_logits: Whether `output` is expected to be a logits tensor. By
            default, we consider that `output` encodes a probability
            distribution.
        label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
            meaning the confidence on label values are relaxed. For example, if
            `0.1`, use `0.1 / num_classes` for non-target labels and
            `0.9 + 0.1 / num_classes` for target labels.
        axis: The axis along which to compute crossentropy (the features
            axis). Defaults to -1.
        reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
            `tf.distribute.Strategy`, except via `Model.compile()` and
            `Model.fit()`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training)
            for more details.
        name: Optional name for the instance.
            Defaults to 'categorical_focal_crossentropy'.

    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="categorical_focal_crossentropy",
    ):
        """Initializes `CategoricalFocalCrossentropy` instance."""
        super().__init__(
            categorical_focal_crossentropy,
            alpha=alpha,
            gamma=gamma,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
        )
        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma

    def get_config(self):
        config = {
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))