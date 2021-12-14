# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf

from iseg.optimizers.adamw import AdamW
from iseg.optimizers.polydecay import WarmUpPolyDecay
from iseg.optimizers.cosinedecay import CosineDecay


def get_optimizer(
    initial_lr=0.007,
    end_lr=0.0,
    epoch_steps=1000,
    train_epoch=30,
    warmup_steps=0,
    warmup_lr=0.003,
    decay_strategy="poly",
    optimizer="sgd",
    sgd_momentum_rate=0.9,
):

    learning_rate = initial_lr
    end_learning_rate = end_lr

    steps = epoch_steps * train_epoch

    if decay_strategy == "poly":

        learning_rate = WarmUpPolyDecay(
            learning_rate,
            steps,
            end_learning_rate=end_learning_rate,
            power=0.9,
            warmup_steps=warmup_steps,
            warmup_learning_rate=warmup_lr,
        )

    elif decay_strategy == "cosine":

        learning_rate = CosineDecay(learning_rate, steps)

    if optimizer == "sgd":
        _optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=sgd_momentum_rate)
    elif optimizer == "adam":
        _optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)
    elif optimizer == "amsgrad":
        _optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
    elif optimizer == "adamw":
        _optimizer = AdamW(weight_decay=0, learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer {optimizer}")

    return _optimizer
