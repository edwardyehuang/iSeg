"""TensorFlow 2 + Keras implementations of FmeasureV2 handlers.

This module provides TensorFlow/Keras implementations of various binary classification
metrics including precision, recall, specificity, dice, IoU, and F-measure.

All handlers inherit from keras.metrics.Metric and can be used in standard Keras workflows.
"""

import abc

import tensorflow as tf
from keras.metrics import Metric

from iseg.metrics.sod.sod_metric_utils import (
    TYPE,
    get_adaptive_threshold,
    safe_divide,
    validate_and_normalize_input,
)


class TFBaseHandler(Metric):
    """Base class for all TensorFlow metric handlers.

    Provides common functionality for calculating various segmentation metrics.
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "base_handler",
        **kwargs,
    ):
        """Initialize the base handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average the metric of each sample or
                calculate the metric of the dataset. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments passed to the parent Metric class.
        """
        super().__init__(name=name, **kwargs)

        self.with_dynamic = with_dynamic
        self.with_adaptive = with_adaptive
        self.with_binary = with_binary
        self.sample_based = sample_based

        # Dynamic results (256 thresholds)
        if with_dynamic:
            self.dynamic_results_sum = self.add_weight(
                name="dynamic_results_sum",
                initializer="zeros",
                shape=(256,),
                dtype=TYPE,
            )
            self.dynamic_count = self.add_weight(
                name="dynamic_count", initializer="zeros", dtype=TYPE
            )

        # Adaptive results
        if with_adaptive:
            self.adaptive_results_sum = self.add_weight(
                name="adaptive_results_sum", initializer="zeros", dtype=TYPE
            )
            self.adaptive_count = self.add_weight(
                name="adaptive_count", initializer="zeros", dtype=TYPE
            )

        # Binary results
        if with_binary:
            if sample_based:
                self.binary_results_sum = self.add_weight(
                    name="binary_results_sum", initializer="zeros", dtype=TYPE
                )
                self.binary_count = self.add_weight(
                    name="binary_count", initializer="zeros", dtype=TYPE
                )
            else:
                self.binary_tp = self.add_weight(
                    name="binary_tp", initializer="zeros", dtype=TYPE
                )
                self.binary_fp = self.add_weight(
                    name="binary_fp", initializer="zeros", dtype=TYPE
                )
                self.binary_tn = self.add_weight(
                    name="binary_tn", initializer="zeros", dtype=TYPE
                )
                self.binary_fn = self.add_weight(
                    name="binary_fn", initializer="zeros", dtype=TYPE
                )

    @abc.abstractmethod
    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate the metric value from confusion matrix components.

        Args:
            tp: True positive count(s)
            fp: False positive count(s)
            tn: True negative count(s)
            fn: False negative count(s)

        Returns:
            Calculated metric value(s)
        """
        pass

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

        FG = tf.cast(tf.math.count_nonzero(gt), TYPE)
        BG = tf.cast(tf.size(gt), TYPE) - FG

        if self.with_dynamic:
            tpfptnfn = self._dynamically_binarizing(pred, gt, FG, BG)
            result = self.compute_metric(**tpfptnfn)
            self.dynamic_results_sum.assign_add(result)
            self.dynamic_count.assign_add(1.0)

        if self.with_adaptive:
            tpfptnfn = self._adaptively_binarizing(pred, gt, FG, BG)
            result = self.compute_metric(**tpfptnfn)
            self.adaptive_results_sum.assign_add(result)
            self.adaptive_count.assign_add(1.0)

        if self.with_binary:
            tpfptnfn = self._get_statistics(pred > 0.5, gt, FG, BG)
            if self.sample_based:
                result = self.compute_metric(**tpfptnfn)
                self.binary_results_sum.assign_add(result)
                self.binary_count.assign_add(1.0)
            else:
                self.binary_tp.assign_add(tpfptnfn["tp"])
                self.binary_fp.assign_add(tpfptnfn["fp"])
                self.binary_tn.assign_add(tpfptnfn["tn"])
                self.binary_fn.assign_add(tpfptnfn["fn"])

    def _get_statistics(
        self, binary: tf.Tensor, gt: tf.Tensor, FG: tf.Tensor, BG: tf.Tensor
    ) -> dict:
        """Calculate TP, FP, TN, FN from binary prediction and ground truth.

        Args:
            binary (tf.Tensor): Binarized prediction containing [0, 1].
            gt (tf.Tensor): Ground truth binarized by threshold.
            FG (tf.Tensor): Number of foreground pixels in gt.
            BG (tf.Tensor): Number of background pixels in gt.

        Returns:
            dict: Dictionary with tp, fp, tn, fn values.
        """
        binary = tf.cast(binary, tf.bool)
        TP = tf.cast(tf.math.count_nonzero(tf.boolean_mask(binary, gt)), TYPE)
        FP = tf.cast(
            tf.math.count_nonzero(tf.boolean_mask(binary, tf.logical_not(gt))), TYPE
        )
        FN = FG - TP
        TN = BG - FP
        return {"tp": TP, "fp": FP, "tn": TN, "fn": FN}

    def _adaptively_binarizing(
        self, pred: tf.Tensor, gt: tf.Tensor, FG: tf.Tensor, BG: tf.Tensor
    ) -> dict:
        """Calculate TP, FP, TN, FN based on adaptive threshold.

        Args:
            pred (tf.Tensor): Prediction normalized in [0, 1].
            gt (tf.Tensor): Ground truth binarized by threshold.
            FG (tf.Tensor): Number of foreground pixels in gt.
            BG (tf.Tensor): Number of background pixels in gt.

        Returns:
            dict: Dictionary with tp, fp, tn, fn values.
        """
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1.0)
        binary = pred >= adaptive_threshold
        return self._get_statistics(binary, gt, FG, BG)

    def _dynamically_binarizing(
        self, pred: tf.Tensor, gt: tf.Tensor, FG: tf.Tensor, BG: tf.Tensor
    ) -> dict:
        """Calculate TP, FP, TN, FN when threshold changes from 0 to 255.

        Args:
            pred (tf.Tensor): Prediction normalized in [0, 1].
            gt (tf.Tensor): Ground truth binarized by threshold.
            FG (tf.Tensor): Number of foreground pixels in gt.
            BG (tf.Tensor): Number of background pixels in gt.

        Returns:
            dict: Dictionary with TPs, FPs, TNs, FNs arrays (256 values each).
        """
        pred_uint8 = tf.cast(pred * 255.0, tf.int32)

        # Get prediction values in foreground and background
        fg_mask = gt
        bg_mask = tf.logical_not(gt)

        fg_pred = tf.boolean_mask(pred_uint8, fg_mask)
        bg_pred = tf.boolean_mask(pred_uint8, bg_mask)

        # Compute histograms
        tp_hist = tf.cast(
            tf.math.bincount(fg_pred, minlength=256, maxlength=256), TYPE
        )
        fp_hist = tf.cast(
            tf.math.bincount(bg_pred, minlength=256, maxlength=256), TYPE
        )

        # Cumulative sum from high to low threshold
        tp_w_thrs = tf.cumsum(tf.reverse(tp_hist, [0]))
        fp_w_thrs = tf.cumsum(tf.reverse(fp_hist, [0]))

        TPs = tp_w_thrs
        FPs = fp_w_thrs
        FNs = FG - TPs
        TNs = BG - FPs

        return {"tp": TPs, "fp": FPs, "tn": TNs, "fn": FNs}

    def result(self) -> dict:
        """Return the computed results.

        Returns:
            dict: Dictionary with dynamic, adaptive, and/or binary results.
        """
        results = {}
        if self.with_dynamic:
            results["dynamic"] = safe_divide(
                self.dynamic_results_sum, self.dynamic_count
            )
        if self.with_adaptive:
            results["adaptive"] = safe_divide(
                self.adaptive_results_sum, self.adaptive_count
            )
        if self.with_binary:
            if self.sample_based:
                results["binary"] = safe_divide(
                    self.binary_results_sum, self.binary_count
                )
            else:
                results["binary"] = self.compute_metric(
                    self.binary_tp, self.binary_fp, self.binary_tn, self.binary_fn
                )
        return results

    def reset_state(self) -> None:
        """Reset the metric state."""
        if self.with_dynamic:
            self.dynamic_results_sum.assign(tf.zeros((256,), dtype=TYPE))
            self.dynamic_count.assign(0.0)
        if self.with_adaptive:
            self.adaptive_results_sum.assign(0.0)
            self.adaptive_count.assign(0.0)
        if self.with_binary:
            if self.sample_based:
                self.binary_results_sum.assign(0.0)
                self.binary_count.assign(0.0)
            else:
                self.binary_tp.assign(0.0)
                self.binary_fp.assign(0.0)
                self.binary_tn.assign(0.0)
                self.binary_fn.assign(0.0)

    def get_config(self) -> dict:
        """Return the config of the metric.

        Returns:
            dict: Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "with_dynamic": self.with_dynamic,
                "with_adaptive": self.with_adaptive,
                "with_binary": self.with_binary,
                "sample_based": self.sample_based,
            }
        )
        return config


class TFIOUHandler(TFBaseHandler):
    """Intersection over Union handler.

    iou = tp / (tp + fp + fn)
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "iou",
        **kwargs,
    ):
        """Initialize IoU handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate IoU from confusion matrix components."""
        return safe_divide(tp, tp + fp + fn)


class TFSpecificityHandler(TFBaseHandler):
    """Specificity handler (True Negative Rate).

    specificity = tn / (tn + fp)
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "specificity",
        **kwargs,
    ):
        """Initialize Specificity handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate specificity from confusion matrix components."""
        return safe_divide(tn, tn + fp)


# Alias for TNR (True Negative Rate)
TFTNRHandler = TFSpecificityHandler


class TFDICEHandler(TFBaseHandler):
    """DICE coefficient handler.

    dice = 2 * tp / (tp + fn + tp + fp)
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "dice",
        **kwargs,
    ):
        """Initialize DICE handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate DICE coefficient from confusion matrix components."""
        return safe_divide(2.0 * tp, tp + fn + tp + fp)


class TFOverallAccuracyHandler(TFBaseHandler):
    """Overall Accuracy handler.

    oa = (tp + tn) / (tp + fp + tn + fn)
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "overall_accuracy",
        **kwargs,
    ):
        """Initialize Overall Accuracy handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate overall accuracy from confusion matrix components."""
        return safe_divide(tp + tn, tp + fp + tn + fn)


class TFKappaHandler(TFBaseHandler):
    """Kappa coefficient handler.

    kappa = (oa - p_) / (1 - p_)
    where p_ = [(tp + fp)(tp + fn) + (tn + fn)(tn + tp)] / (tp + fp + tn + fn)^2
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "kappa",
        **kwargs,
    ):
        """Initialize Kappa handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate Kappa coefficient from confusion matrix components."""
        # Overall accuracy
        total = tp + fp + tn + fn
        oa = safe_divide(tp + tn, total)

        # Hypothetical probability
        hpy_p = safe_divide(
            (tp + fp) * (tp + fn) + (tn + fn) * (tn + tp),
            tf.square(total),
        )

        return safe_divide(oa - hpy_p, 1.0 - hpy_p)


class TFPrecisionHandler(TFBaseHandler):
    """Precision handler.

    precision = tp / (tp + fp)
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "precision",
        **kwargs,
    ):
        """Initialize Precision handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate precision from confusion matrix components."""
        return safe_divide(tp, tp + fp)


class TFRecallHandler(TFBaseHandler):
    """Recall handler (True Positive Rate / Sensitivity).

    recall = tp / (tp + fn)
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "recall",
        **kwargs,
    ):
        """Initialize Recall handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate recall from confusion matrix components."""
        return safe_divide(tp, tp + fn)


# Aliases
TFTPRHandler = TFRecallHandler
TFSensitivityHandler = TFRecallHandler


class TFFPRHandler(TFBaseHandler):
    """False Positive Rate handler.

    fpr = fp / (tn + fp)
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "fpr",
        **kwargs,
    ):
        """Initialize FPR handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate false positive rate from confusion matrix components."""
        return safe_divide(fp, tn + fp)


class TFBERHandler(TFBaseHandler):
    """Balance Error Rate handler.

    ber = 1 - 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        name: str = "ber",
        **kwargs,
    ):
        """Initialize BER handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate balanced error rate from confusion matrix components."""
        fg = tp + fn
        bg = tn + fp
        fg_rate = safe_divide(tp, fg)
        bg_rate = safe_divide(tn, bg)
        return 1.0 - 0.5 * (fg_rate + bg_rate)


class TFFmeasureHandler(TFBaseHandler):
    """F-measure handler.

    fmeasure = (beta + 1) * precision * recall / (beta * precision + recall)
    """

    def __init__(
        self,
        with_dynamic: bool,
        with_adaptive: bool,
        *,
        with_binary: bool = False,
        sample_based: bool = True,
        beta: float = 0.3,
        name: str = "fmeasure",
        **kwargs,
    ):
        """Initialize F-measure handler.

        Args:
            with_dynamic (bool): Record dynamic results for max/avg/curve versions.
            with_adaptive (bool): Record adaptive results for adp version.
            with_binary (bool): Record binary results for binary version.
            sample_based (bool): Whether to average metric per sample. Defaults to True.
            beta (float): β^2 in F-measure. Defaults to 0.3.
            name (str): Name of the metric.
            **kwargs: Additional arguments.
        """
        super().__init__(
            with_dynamic=with_dynamic,
            with_adaptive=with_adaptive,
            with_binary=with_binary,
            sample_based=sample_based,
            name=name,
            **kwargs,
        )
        self.beta = tf.constant(beta, dtype=TYPE)

    def compute_metric(
        self, tp: tf.Tensor, fp: tf.Tensor, tn: tf.Tensor, fn: tf.Tensor
    ) -> tf.Tensor:
        """Calculate F-measure from confusion matrix components."""
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        return safe_divide(
            (self.beta + 1.0) * precision * recall, self.beta * precision + recall
        )

    def get_config(self) -> dict:
        """Return the config of the metric."""
        config = super().get_config()
        config.update({"beta": float(self.beta.numpy())})
        return config


class TFFmeasureV2(Metric):
    """Enhanced F-measure evaluator with support for multiple metric handlers.

    This class provides a flexible framework for computing various binary classification
    metrics including precision, recall, specificity, dice, IoU, and F-measure.
    It supports dynamic thresholding, adaptive thresholding, and binary evaluation modes.
    """

    def __init__(
        self, metric_handlers: dict = None, name: str = "fmeasure_v2", **kwargs
    ):
        """Initialize the FmeasureV2 evaluator.

        Args:
            metric_handlers (dict, optional): Handlers of different metrics. Defaults to None.
            name (str): Name of the metric.
            **kwargs: Additional arguments passed to the parent Metric class.
        """
        super().__init__(name=name, **kwargs)
        self._metric_handlers = metric_handlers if metric_handlers else {}

    def add_handler(self, handler_name: str, metric_handler: TFBaseHandler) -> None:
        """Add a metric handler to the evaluator.

        Args:
            handler_name (str): Name identifier for the metric handler.
            metric_handler: Handler instance that computes the specific metric.
        """
        self._metric_handlers[handler_name] = metric_handler

    def update_state(
        self, pred: tf.Tensor, gt: tf.Tensor, normalize: bool = True
    ) -> None:
        """Update the metric state with a new prediction-ground truth pair.

        Args:
            pred (tf.Tensor): Prediction tensor (grayscale image).
            gt (tf.Tensor): Ground truth tensor (grayscale image).
            normalize (bool): Whether to normalize the input data. Defaults to True.

        Raises:
            ValueError: If no metric handlers have been added.
        """
        if not self._metric_handlers:
            raise ValueError("Please add your metric handler before using `update_state()`.")

        for handler_name, handler in self._metric_handlers.items():
            handler.update_state(pred, gt, normalize)

    def result(self) -> dict:
        """Return the results of all metric handlers.

        Returns:
            dict: Dictionary with results from each metric handler.
        """
        results = {}
        for handler_name, handler in self._metric_handlers.items():
            results[handler_name] = handler.result()
        return results

    def reset_state(self) -> None:
        """Reset the state of all metric handlers."""
        for handler_name, handler in self._metric_handlers.items():
            handler.reset_state()

    def get_config(self) -> dict:
        """Return the config of the metric.

        Returns:
            dict: Configuration dictionary.
        """
        return super().get_config()