# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf


class CoreResult(object):
    def __init__(self, loss_rates=1.0, use_ohem=False, name=None):
        super(CoreResult, self).__init__()

        self.name = name
        self.use_ohem = use_ohem
        self.loss_rates = loss_rates
