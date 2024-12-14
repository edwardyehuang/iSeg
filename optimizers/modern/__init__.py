from iseg.utils.version_utils import is_keras3

if is_keras3():
    from iseg.optimizers.modern_k3.sgd import SGD_EXT as SGD
    from iseg.optimizers.modern_k3.adamw import AdamW_EXT as AdamW
else:
    from .sgd import SGD_EXT as SGD
    from  .adamw import AdamW_EXT as AdamW