# ====================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ====================================================================

import tensorflow as tf


def default_if_not(value, default_value):

    return default_value if not value else value


def to_2d_tuple (x):

    if x is None:
        return None

    if isinstance(x, list):
        x = tuple(x)
    
    if not isinstance(x, tuple):
        x = (x, x)

    return x