# ===================================================================
# MIT License
# Copyright (c) 2024 edwardyehuang (https://github.com/edwardyehuang)
# ===================================================================

import os, sys
import h5py

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

sys.path.insert(1, rootpath)

import tensorflow as tf

from iseg.layers.dcn_v3.dcn_v3 import DeformableConvolutionV3

def test_dcnv3():

    x = tf.ones(shape=(1, 17, 17, 64), dtype=tf.float32)

    dcnv3 = DeformableConvolutionV3(
        filters=64,
        kernel_size=3,
        depthwise_kernel_size=None,
        strides=1,
        padding="SAME",
        dilation_rate=1,
        groups=4,
        offset_scale=1.0,
        center_feature_scale=True,
    )

    y = dcnv3(x)

    print(y)




if __name__ == "__main__":

    test_dcnv3()