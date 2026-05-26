from iseg.utils.version_utils import is_keras3

if is_keras3():
    from iseg.optimizers.modern_k3.sgd import SGD_EXT as SGD
    from iseg.optimizers.modern_k3.adamw import AdamW_EXT as AdamW
    from iseg.optimizers.modern_k3.muon import Muon_EXT as Muon
    from iseg.optimizers.modern_k3.lion import Lion_EXT as Lion
else:
    from .sgd import SGD_EXT as SGD
    from .adamw import AdamW_EXT as AdamW
    from .muon import Muon_EXT as Muon
    from .lion import Lion_EXT as Lion