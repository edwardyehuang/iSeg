# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


class VisualizationTensorRecord(object):
    def __init__(self, name: str, tensor: tf.Tensor, info: str = "") -> None:
        super().__init__()

        tensor = tf.stop_gradient(tensor)

        self.name = name
        self.tensor = tensor
        self.info = info


class VisualizationManager(object):

    __recording = False

    def __init__(self):
        super().__init__()
        self.records = dict()

    @property
    def recording(self):
        return self.__recording

    @recording.setter
    def recording(self, value):
        self.__recording = value

    def add_record(self, record: VisualizationTensorRecord):

        name = record.name
        self.records[name] = record

    def easy_add(self, tensor: tf.Tensor, name: str = "", info: str = ""):

        if name is None or name == "":
            name = tensor.name

        record = VisualizationTensorRecord(name=name, tensor=tensor, info=info)

        self.add_record(record)

    def clear(self):
        self.records.clear()


__vis_manager = VisualizationManager()


def get_visualization_manager():
    return __vis_manager
