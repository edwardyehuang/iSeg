# iSeg : A TensorFlow 2 Library for Semantic Segmentation

## News

January-03-2023: Add support of [ConvNeXtV2](https://arxiv.org/abs/2301.00808).

November-12-2022: Add support of [MOAT](https://arxiv.org/abs/2210.01820).

September-7-2022: Drop support for TensorFlow < 2.10

March-16-2022: We add an example project in [here](https://github.com/edwardyehuang/CAR), which is the offical implementation of the paper [CAR: Class-aware Regularizations for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880514.pdf)

March-16-2022: The backbone weights are now available in [here](backbones/README.md), we will add more in future.


<img src="demo.png" width=450>

## Features
### Backbone support

- [x] Modern ResNet
- [x] Xception-65
- [x] MobileNetV2
- [x] EfficientNetV1
- [x] Feature Pyramid Network
- [x] HRNet
- [x] Swin Transformer (Inputs size free)
- [x] MobileNetV2
- [x] ConvNeXt
- [x] MOAT
- [x] ConvNeXtV2

Weights can be downloaded in [here](backbones/README.md)

### Other features
- [x] Mixed precision training and inference
- [x] Fully deterministic result (100%, see https://github.com/NVIDIA/framework-determinism)
- [x] Training and inference on GPU <= 8
- [x] Training and inference on TPU
- [x] Typical image augmentation
- [x] Support for Windows 10/11
- [x] Support for Windows WSL2
- [ ] One click/out of box deployment for industry (Comming soon)

## Requirements

* TensorFlow >= 2.10 (For iseg <= 0.04, we support TensorFlow >= 2.4)
* Mixed precision only supports GPU architectures after Volta (included).

## Installation (Conda)

```
conda create -n tf210 python=3.8 tensorflow-gpu=2.10 pillow tqdm -c conda-forge
```