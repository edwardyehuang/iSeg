# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from datetime import datetime
from iseg.modelhelper import ModelHelper

'''
Print time at the end of each epoch is useful to determine 
whether the training is stuck (e.g., GPU issue) when the 
user does not fully access system info.
'''
class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):

        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        
        print(f"System time : {datetime.now()}")
