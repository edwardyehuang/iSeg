# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

from distutils.version import LooseVersion
import tensorflow as tf

from iseg.optimizers.adamw import AdamW
from iseg.optimizers.polydecay import WarmUpPolyDecay
from iseg.optimizers.cosinedecay import CosineDecay

def get_optimizer(
    distribute_strategy,
    initial_lr=0.007,
    end_lr=0.0,
    epoch_steps=1000,
    train_epoch=30,
    warmup_steps=0,
    warmup_lr=0.0,
    decay_strategy="poly",
    optimizer="sgd",
    sgd_momentum_rate=0.9,
    adamw_weight_decay=0.0001,
    clipnorm=None,
):

    kwargs = {
        "distribute_strategy": distribute_strategy,
        "initial_lr": initial_lr,
        "end_lr": end_lr,
        "epoch_steps": epoch_steps,
        "train_epoch": train_epoch,
        "warmup_steps": warmup_steps,
        "warmup_lr": warmup_lr,
        "decay_strategy": decay_strategy,
        "optimizer": optimizer,
        "sgd_momentum_rate": sgd_momentum_rate,
        "adamw_weight_decay": adamw_weight_decay,
        "clipnorm": clipnorm,
    }

    print("Optimizer info : **********************")
    print(kwargs)

    keys = kwargs.keys()

    max_list_size = 0

    for key in keys:
        value = kwargs[key]

        if isinstance(value, list) or isinstance(value, tuple):
            value = list(value)
            list_size = len(value)

            kwargs[key] = value # Make sure tuple->list is saved

            assert list_size > 0

            if list_size == 1:
                kwargs[key] = value[0]
            elif list_size >= max_list_size:
                max_list_size = list_size
            else:
                raise ValueError(
                    f"kwargs for optimizer must be scaler or list/tuple with same length, found ({list_size} vs {max_list_size})"
                )

    if max_list_size <= 1:
        return __get_optimizer(**kwargs)

    for key in keys:
        value = kwargs[key]

        if isinstance(value, list):
            list_size = len(value)
            assert (list_size == max_list_size,
                f"kwargs for optimizer must be scaler or list/tuple with same length, found ({list_size} vs {max_list_size})")

    optimizer_list = []

    for i in range(max_list_size):

        sub_kwargs = {}

        for key in keys:
            value = kwargs[key]
        
            if isinstance(value, list):
                value = value[i]

            sub_kwargs[key] = value

        optimizer_list += [__get_optimizer(**sub_kwargs)]

    return optimizer_list
    


def __get_optimizer(
    distribute_strategy,
    initial_lr=0.007,
    end_lr=0.0,
    epoch_steps=1000,
    train_epoch=30,
    warmup_steps=0,
    warmup_lr=0.003,
    decay_strategy="poly",
    optimizer="sgd",
    sgd_momentum_rate=0.9,
    adamw_weight_decay=0.0001,
    clipnorm=None,
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

    with distribute_strategy.scope():
        if LooseVersion(tf.version.VERSION) < LooseVersion("2.10.0"):
            print("TensorFlow version < 2.10, use legacy optimizer")

            if optimizer == "sgd":
                _optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=sgd_momentum_rate)
            elif optimizer == "adam":
                _optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)
            elif optimizer == "amsgrad":
                _optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
            elif optimizer == "adamw":
                _optimizer = AdamW(weight_decay=adamw_weight_decay, learning_rate=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer {optimizer}")
        else:
            if optimizer == "sgd":
                _optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate, momentum=sgd_momentum_rate, clipnorm=clipnorm)
            elif optimizer == "adam":
                _optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=learning_rate, amsgrad=False, clipnorm=clipnorm)
            elif optimizer == "amsgrad":
                _optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=learning_rate, amsgrad=True, clipnorm=clipnorm)
            elif optimizer == "adamw":
                _optimizer = tf.keras.optimizers.experimental.AdamW(weight_decay=adamw_weight_decay, learning_rate=learning_rate, clipnorm=clipnorm)
            else:
                raise ValueError(f"Unsupported optimizer {optimizer}")

        return _optimizer
