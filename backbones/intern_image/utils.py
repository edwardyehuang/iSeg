# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import tensorflow as tf

def extract_qkv (x):

    if isinstance(x, tuple):
        x = list(x)
    if not isinstance(x, list):
        x = [x]

    assert len(x) > 0, 'inputs must not be empty.'

    q = x[0]

    if len(x) == 1:
        k = v = q
    elif len(x) == 2:
        k = v = x[1]
    elif len(x) == 3:
        k = x[1]
        v = x[2]
    else:
        raise ValueError('inputs length must be 1, 2 or 3.')

    return q, k, v