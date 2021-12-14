# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.metrics.seg_metric_wrapper import SegMetricWrapper
from iseg.metrics.mean_iou import MeanIOU


class SegMetricBuilder:
    def __init__(self, num_class, ignore_label):

        self.num_class = num_class
        self.ignore_label = ignore_label

        self.__metrics = []

    def add(self, prefix="", use_iou=True, pre_compute_fn=None):

        metrics_list = []

        if prefix is None:
            prefix = ""

        if prefix != "":
            prefix = prefix + "_"

        if use_iou:
            iou_metric = SegMetricWrapper(
                MeanIOU(self.num_class), num_class=self.num_class, ignore_label=self.ignore_label, name=prefix + "IOU"
            )

            iou_metric.add_pre_compute_fn(pre_compute_fn)

            metrics_list.append(iou_metric)

        self.__metrics.append(metrics_list)

    @property
    def metrics(self):
        return self.__metrics
