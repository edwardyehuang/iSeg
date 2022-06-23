# iSeg : A TensorFlow 2 Library for Semantic Segmentation

## News

March-16-2022: We add an example project in [here](https://github.com/edwardyehuang/CAR), which is the re-implementation of the paper [CAR: Class-aware Regularizations for Semantic Segmentation](https://arxiv.org/pdf/2203.07160.pdf)

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

Weights can be downloaded in [here](backbones/README.md)

### Other features
- [x] Mixed precision training and inference
- [x] Fully Deterministic result (100%, see https://github.com/NVIDIA/framework-determinism)
- [x] Training and inference on GPU <= 8
- [x] Training and inference on TPU
- [x] Typical image augmentation
- [x] Support for Windows 10/11
- [x] Support for Windows WSL2
- [ ] One click/out of box deployment for industry (Plan mid 2022)

## Requirements

* TensorFlow >= 2.4, <= 2.10 (2.3 is fine for TPU users), we recommend to use >= 2.8 for [determinism](https://github.com/NVIDIA/framework-determinism).
* TensorFlow >= 2.8 for Apple M1 users.
* Mixed precision only supports GPU architectures after Volta (included).
