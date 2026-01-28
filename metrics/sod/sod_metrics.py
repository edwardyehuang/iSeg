"""TensorFlow 2 + Keras implementations of SOD (Salient Object Detection) metrics.

This module provides TensorFlow/Keras implementations of standard SOD evaluation metrics
including MAE, S-measure, E-measure, F-measure, Weighted F-measure, and HCE.

All metrics inherit from keras.metrics.Metric and can be used in standard Keras workflows.
"""

import tensorflow as tf
from keras.metrics import Metric

from iseg.metrics.sod.sod_metric_utils import (
    EPS,
    TYPE,
    get_adaptive_threshold,
    safe_divide,
    tf_convolve2d,
    tf_count_nonzero,
    tf_count_polygon_control_points,
    tf_distance_transform_edt,
    tf_filter_conditional_boundary,
    tf_find_contours,
    tf_gaussian_kernel,
    tf_morphology_dilate,
    tf_morphology_erode,
    tf_skeletonize,
    validate_and_normalize_input,
)


class TFMAEMetric(Metric):
    """Mean Absolute Error metric for salient object detection.

    Computes the MAE between predicted saliency maps and ground truth masks.

    ```
    @inproceedings{MAE,
        title={Saliency filters: Contrast based filtering for salient region detection},
        author={Perazzi, Federico and Krähenb\\"uhl, Philipp and Pritch, Yael and Hornung, Alexander},
        booktitle=CVPR,
        pages={733--740},
        year={2012}
    }
    ```
    """

    def __init__(self, name: str = "mae", **kwargs):
        """Initialize the MAE metric.

        Args:
            name (str): Name of the metric.
            **kwargs: Additional arguments passed to the parent Metric class.
        """
        super().__init__(name=name, **kwargs)
        self.mae_sum = self.add_weight(name="mae_sum", initializer="zeros", dtype=TYPE)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=TYPE)

    def update_state(
        self, pred: tf.Tensor, gt: tf.Tensor, normalize: bool = True
    ) -> None:
        """Update the metric state with a new prediction-ground truth pair.

        Args:
            pred (tf.Tensor): Prediction tensor (grayscale image).
            gt (tf.Tensor): Ground truth tensor (grayscale image).
            normalize (bool): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        mae = self._cal_mae(pred, gt)
        self.mae_sum.assign_add(mae)
        self.count.assign_add(1.0)

    def _cal_mae(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate the mean absolute error.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: MAE value.
        """
        gt = tf.cast(gt, TYPE)
        return tf.reduce_mean(tf.abs(pred - gt))

    def result(self) -> tf.Tensor:
        """Return the computed MAE.

        Returns:
            tf.Tensor: Mean absolute error.
        """
        return safe_divide(self.mae_sum, self.count)

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.mae_sum.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict:
        """Return the config of the metric.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        return config


class TFSmeasureMetric(Metric):
    """S-measure metric for salient object detection.

    S-measure evaluates foreground maps by considering both object-aware and region-aware
    structural similarity between prediction and ground truth.

    ```
    @inproceedings{Smeasure,
        title={Structure-measure: A new way to eval foreground maps},
        author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
        booktitle=ICCV,
        pages={4548--4557},
        year={2017}
    }
    ```
    """

    def __init__(self, alpha: float = 0.5, name: str = "sm", **kwargs):
        """Initialize S-measure (Structure-measure) metric.

        Args:
            alpha (float): Weight for balancing the object score and the region score.
                Higher values give more weight to object-level similarity.
                Valid range: [0, 1]. Defaults to 0.5 for equal weighting.
            name (str): Name of the metric.
            **kwargs: Additional arguments passed to the parent Metric class.
        """
        super().__init__(name=name, **kwargs)
        self.alpha = tf.constant(alpha, dtype=TYPE)
        self.sm_sum = self.add_weight(name="sm_sum", initializer="zeros", dtype=TYPE)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=TYPE)

    def update_state(
        self, pred: tf.Tensor, gt: tf.Tensor, normalize: bool = True
    ) -> None:
        """Update the metric state with a new prediction-ground truth pair.

        Args:
            pred (tf.Tensor): Prediction tensor (grayscale image).
            gt (tf.Tensor): Ground truth tensor (grayscale image).
            normalize (bool): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        sm = self._cal_sm(pred, gt)
        self.sm_sum.assign_add(sm)
        self.count.assign_add(1.0)

    def _cal_sm(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate the S-measure (Structure-measure) score.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: S-measure score in range [0, 1].
        """
        gt_float = tf.cast(gt, TYPE)
        y = tf.reduce_mean(gt_float)

        def all_background():
            return 1.0 - tf.reduce_mean(pred)

        def all_foreground():
            return tf.reduce_mean(pred)

        def mixed():
            object_score = self._object(pred, gt) * self.alpha
            region_score = self._region(pred, gt) * (1.0 - self.alpha)
            return tf.maximum(0.0, object_score + region_score)

        sm = tf.case(
            [(tf.equal(y, 0.0), all_background), (tf.equal(y, 1.0), all_foreground)],
            default=mixed,
        )
        return sm

    def _s_object(self, x: tf.Tensor) -> tf.Tensor:
        """Calculate object-aware score for a region.

        Args:
            x (tf.Tensor): Input region data.

        Returns:
            tf.Tensor: Object-aware similarity score.
        """
        mean = tf.reduce_mean(x)
        std = tf.math.reduce_std(x)
        score = 2.0 * mean / (tf.square(mean) + 1.0 + std + EPS)
        return score

    def _object(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate the object-level structural similarity score.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: Object-level similarity score.
        """
        gt_float = tf.cast(gt, TYPE)
        gt_mean = tf.reduce_mean(gt_float)

        # Foreground region
        fg_pred = tf.boolean_mask(pred, gt)
        fg_score = self._s_object(fg_pred) * gt_mean

        # Background region
        bg_mask = tf.logical_not(gt)
        bg_pred = tf.boolean_mask(1.0 - pred, bg_mask)
        bg_score = self._s_object(bg_pred) * (1.0 - gt_mean)

        return fg_score + bg_score

    def _region(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate the region-level structural similarity score.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: Region-level similarity score.
        """
        gt_float = tf.cast(gt, TYPE)
        shape = tf.shape(gt)
        h = tf.cast(shape[0], TYPE)
        w = tf.cast(shape[1], TYPE)
        area = h * w

        # Calculate the centroid coordinate of the foreground
        num_fg = tf_count_nonzero(gt)

        def compute_centroid_empty():
            return tf.round(h / 2.0), tf.round(w / 2.0)

        def compute_centroid_nonempty():
            # Get indices of foreground pixels
            indices = tf.where(gt)
            indices_float = tf.cast(indices, TYPE)
            cy = tf.round(tf.reduce_mean(indices_float[:, 0]))
            cx = tf.round(tf.reduce_mean(indices_float[:, 1]))
            return cy, cx

        cy, cx = tf.cond(
            tf.equal(num_fg, 0),
            compute_centroid_empty,
            compute_centroid_nonempty,
        )

        # Add 1 for MATLAB compatibility
        cy = tf.cast(cy, tf.int32) + 1
        cx = tf.cast(cx, tf.int32) + 1
        h_int = tf.cast(h, tf.int32)
        w_int = tf.cast(w, tf.int32)

        cy_f = tf.cast(cy, TYPE)
        cx_f = tf.cast(cx, TYPE)

        # Calculate weights for each quadrant
        w_lt = cy_f * cx_f / area
        w_rt = cy_f * (w - cx_f) / area
        w_lb = (h - cy_f) * cx_f / area
        w_rb = 1.0 - w_lt - w_rt - w_lb

        # Calculate SSIM for each quadrant
        score_lt = self._ssim(pred[:cy, :cx], gt_float[:cy, :cx]) * w_lt
        score_rt = self._ssim(pred[:cy, cx:w_int], gt_float[:cy, cx:w_int]) * w_rt
        score_lb = self._ssim(pred[cy:h_int, :cx], gt_float[cy:h_int, :cx]) * w_lb
        score_rb = (
            self._ssim(pred[cy:h_int, cx:w_int], gt_float[cy:h_int, cx:w_int]) * w_rb
        )

        return score_lt + score_rt + score_lb + score_rb

    def _ssim(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate the SSIM (Structural Similarity Index) score.

        Args:
            pred (tf.Tensor): Prediction region.
            gt (tf.Tensor): Ground truth region.

        Returns:
            tf.Tensor: SSIM score in range [0, 1].
        """
        shape = tf.shape(pred)
        N = tf.cast(shape[0] * shape[1], TYPE)

        x = tf.reduce_mean(pred)
        y = tf.reduce_mean(gt)

        sigma_x = tf.reduce_sum(tf.square(pred - x)) / (N - 1.0)
        sigma_y = tf.reduce_sum(tf.square(gt - y)) / (N - 1.0)
        sigma_xy = tf.reduce_sum((pred - x) * (gt - y)) / (N - 1.0)

        alpha = 4.0 * x * y * sigma_xy
        beta = (tf.square(x) + tf.square(y)) * (sigma_x + sigma_y)

        def alpha_nonzero():
            return alpha / (beta + EPS)

        def alpha_zero_beta_zero():
            return tf.constant(1.0, dtype=TYPE)

        def alpha_zero_beta_nonzero():
            return tf.constant(0.0, dtype=TYPE)

        score = tf.case(
            [
                (tf.not_equal(alpha, 0.0), alpha_nonzero),
                (
                    tf.logical_and(tf.equal(alpha, 0.0), tf.equal(beta, 0.0)),
                    alpha_zero_beta_zero,
                ),
            ],
            default=alpha_zero_beta_nonzero,
        )
        return score

    def result(self) -> tf.Tensor:
        """Return the computed S-measure.

        Returns:
            tf.Tensor: Mean S-measure score.
        """
        return safe_divide(self.sm_sum, self.count)

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.sm_sum.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict:
        """Return the config of the metric.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({"alpha": float(self.alpha.numpy())})
        return config


class TFEmeasureMetric(Metric):
    """E-measure metric for salient object detection.

    E-measure assesses binary foreground map quality by measuring the alignment
    between prediction and ground truth using an enhanced alignment matrix.

    ```
    @inproceedings{Emeasure,
        title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
        author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and
                Ming-Ming {Cheng} and Ali {Borji}",
        booktitle=IJCAI,
        pages="698--704",
        year={2018}
    }
    ```
    """

    def __init__(self, name: str = "em", **kwargs):
        """Initialize E-measure (Enhanced-alignment Measure) metric.

        Args:
            name (str): Name of the metric.
            **kwargs: Additional arguments passed to the parent Metric class.
        """
        super().__init__(name=name, **kwargs)
        # For adaptive E-measure
        self.adaptive_em_sum = self.add_weight(
            name="adaptive_em_sum", initializer="zeros", dtype=TYPE
        )
        self.adaptive_count = self.add_weight(
            name="adaptive_count", initializer="zeros", dtype=TYPE
        )
        # For changeable E-measure (256 thresholds)
        self.changeable_em_sum = self.add_weight(
            name="changeable_em_sum",
            initializer="zeros",
            shape=(256,),
            dtype=TYPE,
        )
        self.changeable_count = self.add_weight(
            name="changeable_count", initializer="zeros", dtype=TYPE
        )

    def update_state(
        self, pred: tf.Tensor, gt: tf.Tensor, normalize: bool = True
    ) -> None:
        """Update the metric state with a new prediction-ground truth pair.

        Args:
            pred (tf.Tensor): Prediction tensor (grayscale image).
            gt (tf.Tensor): Ground truth tensor (grayscale image).
            normalize (bool): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        self.gt_fg_numel = tf_count_nonzero(gt)
        shape = tf.shape(gt)
        self.gt_size = tf.cast(shape[0] * shape[1], tf.int64)

        # Calculate adaptive E-measure
        adaptive_em = self._cal_adaptive_em(pred, gt)
        self.adaptive_em_sum.assign_add(adaptive_em)
        self.adaptive_count.assign_add(1.0)

        # Calculate changeable E-measure
        changeable_ems = self._cal_changeable_em(pred, gt)
        self.changeable_em_sum.assign_add(changeable_ems)
        self.changeable_count.assign_add(1.0)

    def _cal_adaptive_em(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate the adaptive E-measure using an adaptive threshold.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: Adaptive E-measure score.
        """
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1.0)
        return self._cal_em_with_threshold(pred, gt, adaptive_threshold)

    def _cal_changeable_em(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate E-measure scores across all thresholds from 0 to 255.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: Array of 256 E-measure scores.
        """
        return self._cal_em_with_cumsumhistogram(pred, gt)

    def _cal_em_with_threshold(
        self, pred: tf.Tensor, gt: tf.Tensor, threshold: tf.Tensor
    ) -> tf.Tensor:
        """Calculate the E-measure for a specific binarization threshold.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.
            threshold (tf.Tensor): Binarization threshold value.

        Returns:
            tf.Tensor: E-measure score for the given threshold.
        """
        binarized_pred = pred >= threshold
        fg_fg_numel = tf_count_nonzero(tf.logical_and(binarized_pred, gt))
        fg_bg_numel = tf_count_nonzero(
            tf.logical_and(binarized_pred, tf.logical_not(gt))
        )

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        def gt_all_bg():
            return tf.cast(bg___numel, TYPE)

        def gt_all_fg():
            return tf.cast(fg___numel, TYPE)

        def gt_mixed():
            parts_numel, combinations = self._generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
            )

            enhanced_matrix_sum = tf.constant(0.0, dtype=TYPE)
            for part_numel, combination in zip(parts_numel, combinations):
                align_matrix_value = (
                    2.0
                    * (combination[0] * combination[1])
                    / (tf.square(combination[0]) + tf.square(combination[1]) + EPS)
                )
                enhanced_matrix_value = tf.square(align_matrix_value + 1.0) / 4.0
                enhanced_matrix_sum = enhanced_matrix_sum + enhanced_matrix_value * tf.cast(
                    part_numel, TYPE
                )
            return enhanced_matrix_sum

        enhanced_matrix_sum = tf.case(
            [
                (tf.equal(self.gt_fg_numel, 0), gt_all_bg),
                (tf.equal(self.gt_fg_numel, self.gt_size), gt_all_fg),
            ],
            default=gt_mixed,
        )

        em = enhanced_matrix_sum / (tf.cast(self.gt_size, TYPE) - 1.0 + EPS)
        return em

    def _cal_em_with_cumsumhistogram(
        self, pred: tf.Tensor, gt: tf.Tensor
    ) -> tf.Tensor:
        """Calculate E-measure using cumulative histogram for all thresholds.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: Array of 256 E-measure scores.
        """
        pred_uint8 = tf.cast(pred * 255.0, tf.int32)

        # Get foreground and background masks
        fg_mask = gt
        bg_mask = tf.logical_not(gt)

        # Get prediction values in foreground and background
        fg_pred = tf.boolean_mask(pred_uint8, fg_mask)
        bg_pred = tf.boolean_mask(pred_uint8, bg_mask)

        # Compute histograms
        fg_fg_hist = tf.cast(
            tf.math.bincount(fg_pred, minlength=256, maxlength=256), TYPE
        )
        fg_bg_hist = tf.cast(
            tf.math.bincount(bg_pred, minlength=256, maxlength=256), TYPE
        )

        # Cumulative sum from high to low threshold
        fg_fg_numel_w_thrs = tf.cumsum(tf.reverse(fg_fg_hist, [0]))
        fg_bg_numel_w_thrs = tf.cumsum(tf.reverse(fg_bg_hist, [0]))

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = tf.cast(self.gt_size, TYPE) - fg___numel_w_thrs

        def gt_all_bg():
            return bg___numel_w_thrs

        def gt_all_fg():
            return fg___numel_w_thrs

        def gt_mixed():
            parts_numel_w_thrs, combinations = self._generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = []
            for part_numel, combination in zip(parts_numel_w_thrs, combinations):
                part_numel = tf.cast(part_numel, TYPE)
                align_matrix_value = (
                    2.0
                    * (combination[0] * combination[1])
                    / (tf.square(combination[0]) + tf.square(combination[1]) + EPS)
                )
                enhanced_matrix_value = tf.square(align_matrix_value + 1.0) / 4.0
                results_parts.append(enhanced_matrix_value * part_numel)
            return tf.reduce_sum(tf.stack(results_parts, axis=0), axis=0)

        enhanced_matrix_sum = tf.case(
            [
                (tf.equal(self.gt_fg_numel, 0), gt_all_bg),
                (tf.equal(self.gt_fg_numel, self.gt_size), gt_all_fg),
            ],
            default=gt_mixed,
        )

        em = enhanced_matrix_sum / (tf.cast(self.gt_size, TYPE) - 1.0 + EPS)
        return em

    def _generate_parts_numel_combinations(
        self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel
    ):
        """Generate the number of elements in each part of the image.

        Args:
            fg_fg_numel: Number of foreground pixels in the foreground region.
            fg_bg_numel: Number of foreground pixels in the background region.
            pred_fg_numel: Number of foreground pixels in the predicted region.
            pred_bg_numel: Number of background pixels in the predicted region.

        Returns:
            tuple: Parts numel and combinations.
        """
        fg_fg_numel = tf.cast(fg_fg_numel, TYPE)
        fg_bg_numel = tf.cast(fg_bg_numel, TYPE)
        pred_fg_numel = tf.cast(pred_fg_numel, TYPE)
        pred_bg_numel = tf.cast(pred_bg_numel, TYPE)
        gt_fg_numel = tf.cast(self.gt_fg_numel, TYPE)
        gt_size = tf.cast(self.gt_size, TYPE)

        bg_fg_numel = gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / gt_size
        mean_gt_value = gt_fg_numel / gt_size

        demeaned_pred_fg_value = 1.0 - mean_pred_value
        demeaned_pred_bg_value = 0.0 - mean_pred_value
        demeaned_gt_fg_value = 1.0 - mean_gt_value
        demeaned_gt_bg_value = 0.0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def result(self) -> dict:
        """Return the computed E-measure results.

        Returns:
            dict: Dictionary with 'adp' (adaptive) and 'curve' (changeable) E-measures.
        """
        adaptive_em = safe_divide(self.adaptive_em_sum, self.adaptive_count)
        changeable_em = safe_divide(self.changeable_em_sum, self.changeable_count)
        return {"adp": adaptive_em, "curve": changeable_em}

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.adaptive_em_sum.assign(0.0)
        self.adaptive_count.assign(0.0)
        self.changeable_em_sum.assign(tf.zeros((256,), dtype=TYPE))
        self.changeable_count.assign(0.0)

    def get_config(self) -> dict:
        """Return the config of the metric.

        Returns:
            dict: Configuration dictionary.
        """
        return super().get_config()


class TFFmeasureMetric(Metric):
    """F-measure metric for salient object detection.

    Computes precision, recall, and F-measure at multiple thresholds,
    supporting both adaptive and dynamic evaluation modes.

    ```
    @inproceedings{Fmeasure,
        title={Frequency-tuned salient region detection},
        author={Achanta, Radhakrishna and Hemami, Sheila and Estrada, Francisco and
                S\\"usstrunk, Sabine},
        booktitle=CVPR,
        number={CONF},
        pages={1597--1604},
        year={2009}
    }
    ```
    """

    def __init__(self, beta: float = 0.3, name: str = "fm", **kwargs):
        """Initialize the F-measure metric.

        Args:
            beta (float): The weight of the precision. Defaults to 0.3.
            name (str): Name of the metric.
            **kwargs: Additional arguments passed to the parent Metric class.
        """
        super().__init__(name=name, **kwargs)
        self.beta = tf.constant(beta, dtype=TYPE)

        # Adaptive F-measure
        self.adaptive_fm_sum = self.add_weight(
            name="adaptive_fm_sum", initializer="zeros", dtype=TYPE
        )
        # Changeable F-measure (257 thresholds)
        self.changeable_fm_sum = self.add_weight(
            name="changeable_fm_sum",
            initializer="zeros",
            shape=(257,),
            dtype=TYPE,
        )
        # Precision and recall curves
        self.precision_sum = self.add_weight(
            name="precision_sum", initializer="zeros", shape=(257,), dtype=TYPE
        )
        self.recall_sum = self.add_weight(
            name="recall_sum", initializer="zeros", shape=(257,), dtype=TYPE
        )
        self.count = self.add_weight(name="count", initializer="zeros", dtype=TYPE)

    def update_state(
        self, pred: tf.Tensor, gt: tf.Tensor, normalize: bool = True
    ) -> None:
        """Update the metric state with a new prediction-ground truth pair.

        Args:
            pred (tf.Tensor): Prediction tensor (grayscale image).
            gt (tf.Tensor): Ground truth tensor (grayscale image).
            normalize (bool): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        # Adaptive F-measure
        adaptive_fm = self._cal_adaptive_fm(pred, gt)
        self.adaptive_fm_sum.assign_add(adaptive_fm)

        # Precision, recall, and changeable F-measure
        precisions, recalls, changeable_fms = self._cal_pr(pred, gt)
        self.precision_sum.assign_add(precisions)
        self.recall_sum.assign_add(recalls)
        self.changeable_fm_sum.assign_add(changeable_fms)

        self.count.assign_add(1.0)

    def _cal_adaptive_fm(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate the adaptive F-measure.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: Adaptive F-measure score.
        """
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1.0)
        binary_prediction = pred >= adaptive_threshold
        area_intersection = tf_count_nonzero(tf.logical_and(binary_prediction, gt))

        def zero_intersection():
            return tf.constant(0.0, dtype=TYPE)

        def nonzero_intersection():
            area_intersection_f = tf.cast(area_intersection, TYPE)
            pre = area_intersection_f / tf.cast(
                tf_count_nonzero(binary_prediction), TYPE
            )
            rec = area_intersection_f / tf.cast(tf_count_nonzero(gt), TYPE)
            return (1.0 + self.beta) * pre * rec / (self.beta * pre + rec)

        adaptive_fm = tf.cond(
            tf.equal(area_intersection, 0),
            zero_intersection,
            nonzero_intersection,
        )
        return adaptive_fm

    def _cal_pr(self, pred: tf.Tensor, gt: tf.Tensor) -> tuple:
        """Calculate precision and recall when threshold varies from 0 to 255.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tuple: (precisions, recalls, changeable_fms)
        """
        pred_uint8 = tf.cast(pred * 255.0, tf.int32)

        # Get foreground and background masks
        fg_mask = gt
        bg_mask = tf.logical_not(gt)

        # Get prediction values in foreground and background
        fg_pred = tf.boolean_mask(pred_uint8, fg_mask)
        bg_pred = tf.boolean_mask(pred_uint8, bg_mask)

        # Compute histograms
        fg_hist = tf.cast(
            tf.math.bincount(fg_pred, minlength=257, maxlength=257), TYPE
        )
        bg_hist = tf.cast(
            tf.math.bincount(bg_pred, minlength=257, maxlength=257), TYPE
        )

        # Cumulative sum from high to low threshold
        fg_w_thrs = tf.cumsum(tf.reverse(fg_hist, [0]))
        bg_w_thrs = tf.cumsum(tf.reverse(bg_hist, [0]))

        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs

        # Handle division by zero
        Ps_safe = tf.where(tf.equal(Ps, 0.0), tf.ones_like(Ps), Ps)
        T = tf.maximum(tf.cast(tf_count_nonzero(gt), TYPE), 1.0)

        precisions = TPs / Ps_safe
        precisions = tf.where(tf.equal(Ps, 0.0), tf.zeros_like(precisions), precisions)
        recalls = TPs / T

        numerator = (1.0 + self.beta) * precisions * recalls
        denominator = tf.where(
            tf.equal(numerator, 0.0),
            tf.ones_like(numerator),
            self.beta * precisions + recalls,
        )
        changeable_fms = numerator / denominator

        return precisions, recalls, changeable_fms

    def result(self) -> dict:
        """Return the computed F-measure results.

        Returns:
            dict: Dictionary with F-measure and PR curve data.
        """
        adaptive_fm = safe_divide(self.adaptive_fm_sum, self.count)
        changeable_fm = safe_divide(self.changeable_fm_sum, self.count)
        precision = safe_divide(self.precision_sum, self.count)
        recall = safe_divide(self.recall_sum, self.count)
        return {
            "fm": {"adp": adaptive_fm, "curve": changeable_fm},
            "pr": {"p": precision, "r": recall},
        }

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.adaptive_fm_sum.assign(0.0)
        self.changeable_fm_sum.assign(tf.zeros((257,), dtype=TYPE))
        self.precision_sum.assign(tf.zeros((257,), dtype=TYPE))
        self.recall_sum.assign(tf.zeros((257,), dtype=TYPE))
        self.count.assign(0.0)

    def get_config(self) -> dict:
        """Return the config of the metric.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({"beta": float(self.beta.numpy())})
        return config


class TFWeightedFmeasureMetric(Metric):
    """Weighted F-measure metric for salient object detection.

    Weighted F-measure considers both pixel dependency and pixel importance
    when evaluating foreground maps.

    ```
    @inproceedings{wFmeasure,
        title={How to eval foreground maps?},
        author={Margolin, Ran and Zelnik-Manor, Lihi and Tal, Ayellet},
        booktitle=CVPR,
        pages={248--255},
        year={2014}
    }
    ```
    """

    def __init__(self, beta: float = 1.0, name: str = "wfm", **kwargs):
        """Initialize Weighted F-measure metric.

        Args:
            beta (float): Weight for balancing precision and recall.
                Defaults to 1 for equal weighting (F1-score).
            name (str): Name of the metric.
            **kwargs: Additional arguments passed to the parent Metric class.
        """
        super().__init__(name=name, **kwargs)
        self.beta = tf.constant(beta, dtype=TYPE)
        self.wfm_sum = self.add_weight(name="wfm_sum", initializer="zeros", dtype=TYPE)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=TYPE)

    def update_state(
        self, pred: tf.Tensor, gt: tf.Tensor, normalize: bool = True
    ) -> None:
        """Update the metric state with a new prediction-ground truth pair.

        Args:
            pred (tf.Tensor): Prediction tensor (grayscale image).
            gt (tf.Tensor): Ground truth tensor (grayscale image).
            normalize (bool): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        # Check if gt is all background
        gt_any = tf.reduce_any(gt)
        wfm = tf.cond(
            gt_any,
            lambda: self._cal_wfm(pred, gt),
            lambda: tf.constant(0.0, dtype=TYPE),
        )
        self.wfm_sum.assign_add(wfm)
        self.count.assign_add(1.0)

    def _cal_wfm(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate the weighted F-measure score.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: Weighted F-measure score.
        """
        gt_float = tf.cast(gt, TYPE)
        gt_bool = tf.cast(gt, tf.bool)
        gt_zero = tf.equal(gt_float, 0.0)

        # Distance transform
        Dst, (Idxt_y, Idxt_x) = tf_distance_transform_edt(gt_zero, return_indices=True)

        # Pixel dependency
        E = tf.abs(pred - gt_float)
        Et = E

        # Et[~gt] = Et[Idxt[0][~gt], Idxt[1][~gt]]
        bg_indices = tf.where(gt_zero)
        bg_y = tf.gather_nd(Idxt_y, bg_indices)
        bg_x = tf.gather_nd(Idxt_x, bg_indices)
        bg_coords = tf.stack([bg_y, bg_x], axis=1)
        Et_bg_values = tf.gather_nd(E, bg_coords)

        # Create updated Et
        Et = tf.tensor_scatter_nd_update(Et, bg_indices, Et_bg_values)

        # Gaussian filter
        K = tf_gaussian_kernel((7, 7), sigma=5.0)
        EA = tf_convolve2d(Et, K, mode="constant", cval=0.0)

        # MIN_E_EA
        MIN_E_EA = tf.where(tf.logical_and(gt_bool, EA < E), EA, E)

        # Pixel importance
        B = tf.where(
            gt_zero,
            2.0 - tf.exp(tf.math.log(0.5) / 5.0 * Dst),
            tf.ones_like(gt_float),
        )
        Ew = MIN_E_EA * B

        # Calculate weighted precision and recall
        TPw = tf.reduce_sum(gt_float) - tf.reduce_sum(tf.boolean_mask(Ew, gt_bool))
        FPw = tf.reduce_sum(tf.boolean_mask(Ew, gt_zero))

        R = 1.0 - tf.reduce_mean(tf.boolean_mask(Ew, gt_bool))
        P = TPw / (TPw + FPw + EPS)

        Q = (1.0 + self.beta) * R * P / (R + self.beta * P + EPS)
        return Q

    def result(self) -> tf.Tensor:
        """Return the computed Weighted F-measure.

        Returns:
            tf.Tensor: Mean weighted F-measure score.
        """
        return safe_divide(self.wfm_sum, self.count)

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.wfm_sum.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict:
        """Return the config of the metric.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({"beta": float(self.beta.numpy())})
        return config


class TFHumanCorrectionEffortMeasure(Metric):
    """Human Correction Effort Measure for Dichotomous Image Segmentation.

    This metric measures the human effort needed to correct segmentation errors.

    ```
    @inproceedings{HumanCorrectionEffortMeasure,
        title = {Highly Accurate Dichotomous Image Segmentation},
        author = {Xuebin Qin and Hang Dai and Xiaobin Hu and Deng-Ping Fan and
                  Ling Shao and Luc Van Gool},
        booktitle = ECCV,
        year = {2022}
    }
    ```

    Note:
        This is a simplified TensorFlow implementation. Some operations like
        cv2.findContours and cv2.approxPolyDP are approximated using TensorFlow
        operations, which may produce slightly different results than the original
        NumPy/OpenCV implementation.
    """

    def __init__(
        self,
        relax: int = 5,
        epsilon: float = 2.0,
        name: str = "hce",
        **kwargs,
    ):
        """Initialize the Human Correction Effort Measure.

        Args:
            relax (int): The number of relaxations. Defaults to 5.
            epsilon (float): The epsilon value for polygon approximation. Defaults to 2.0.
            name (str): Name of the metric.
            **kwargs: Additional arguments passed to the parent Metric class.
        """
        super().__init__(name=name, **kwargs)
        self.relax = relax
        self.epsilon = epsilon
        self.hce_sum = self.add_weight(name="hce_sum", initializer="zeros", dtype=TYPE)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=TYPE)

    def update_state(
        self, pred: tf.Tensor, gt: tf.Tensor, normalize: bool = True
    ) -> None:
        """Update the metric state with a new prediction-ground truth pair.

        Args:
            pred (tf.Tensor): Prediction tensor (grayscale image).
            gt (tf.Tensor): Ground truth tensor (grayscale image).
            normalize (bool): Whether to normalize the input data. Defaults to True.
        """
        pred, gt = validate_and_normalize_input(pred, gt, normalize)

        hce = self._cal_hce(pred, gt)
        self.hce_sum.assign_add(hce)
        self.count.assign_add(1.0)

    def _cal_hce(self, pred: tf.Tensor, gt: tf.Tensor) -> tf.Tensor:
        """Calculate the Human Correction Effort (HCE) for a prediction-ground truth pair.

        Full implementation following the original algorithm with contour finding and
        polygon control point counting using Ramer-Douglas-Peucker approximation.

        Args:
            pred (tf.Tensor): Normalized prediction map with values in [0, 1].
            gt (tf.Tensor): Binary ground truth mask.

        Returns:
            tf.Tensor: The HCE value.
        """
        # Skeletonize ground truth
        gt_skeleton = tf_skeletonize(gt)

        # Binarize prediction
        pred_binary = pred > 0.5

        # Compute TP, FP, FN
        union = tf.logical_or(gt, pred_binary)
        TP = tf.logical_and(gt, pred_binary)
        FP = tf.logical_xor(pred_binary, TP)
        FN = tf.logical_xor(gt, TP)

        # Relax the union using disk-shaped kernel (approximated with 3x3)
        eroded_union = tf_morphology_erode(
            tf.cast(union, TYPE), kernel_size=3, iterations=self.relax
        )
        eroded_union = eroded_union > 0

        # Get relaxed FP regions
        FP_ = tf.logical_and(FP, eroded_union)
        for _ in range(self.relax):
            FP_dilated = tf_morphology_dilate(
                tf.cast(FP_, TYPE), kernel_size=3, iterations=1
            )
            FP_ = tf.logical_and(FP_dilated > 0, tf.logical_not(gt))
        FP_ = tf.logical_and(FP, FP_)

        # Get relaxed FN regions
        FN_ = tf.logical_and(FN, eroded_union)
        for _ in range(self.relax):
            FN_dilated = tf_morphology_dilate(
                tf.cast(FN_, TYPE), kernel_size=3, iterations=1
            )
            FN_ = tf.logical_and(FN_dilated > 0, tf.logical_not(pred_binary))
        FN_ = tf.logical_and(FN, FN_)

        # Preserve structural components of FN
        FN_ = tf.logical_or(
            FN_, tf.logical_xor(gt_skeleton, tf.logical_and(TP, gt_skeleton))
        )

        # Find contours and filter boundaries for FP regions
        contours_FP = tf_find_contours(FP_)
        condition_FP = tf.logical_or(TP, FN_)
        bdies_FP, indep_cnt_FP = tf_filter_conditional_boundary(
            contours_FP, FP_, condition_FP
        )

        # Find contours and filter boundaries for FN regions
        contours_FN = tf_find_contours(FN_)
        condition_FN = tf.logical_not(
            tf.logical_or(tf.logical_or(TP, FP_), FN_)
        )
        bdies_FN, indep_cnt_FN = tf_filter_conditional_boundary(
            contours_FN, FN_, condition_FN
        )

        # Count polygon control points using RDP approximation
        poly_FP_point_cnt = tf_count_polygon_control_points(bdies_FP, self.epsilon)
        poly_FN_point_cnt = tf_count_polygon_control_points(bdies_FN, self.epsilon)

        # Sum all contributions
        total = (
            tf.cast(poly_FP_point_cnt, TYPE) +
            tf.cast(indep_cnt_FP, TYPE) +
            tf.cast(poly_FN_point_cnt, TYPE) +
            tf.cast(indep_cnt_FN, TYPE)
        )

        return total

    def result(self) -> tf.Tensor:
        """Return the computed HCE.

        Returns:
            tf.Tensor: Mean HCE value.
        """
        return safe_divide(self.hce_sum, self.count)

    def reset_state(self) -> None:
        """Reset the metric state."""
        self.hce_sum.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> dict:
        """Return the config of the metric.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update({"relax": self.relax, "epsilon": self.epsilon})
        return config