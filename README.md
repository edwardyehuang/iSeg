# iSeg : A TensorFlow 2 Library for Semantic Segmentation

## News

March-16-2022: We add an example project in [here](https://github.com/edwardyehuang/CAR), which is the re-implementation of the paper [CAR: Class-aware Regularizations for Semantic Segmentation](https://arxiv.org/pdf/2203.07160.pdf)

March-16-2022: The backbone weights are now available in [here](backbones/weights_download.md), we will add more in future.


<img src="demo.png" width=450>

## Features
### Backbone support

- [x] Modern ResNet
- [x] Xception-65
- [x] EfficientNetV1
- [ ] Vision Transformer (WIP)
- [ ] MLP Mixer (WIP)
- [ ] SegFormer
- [x] Feature Pyramid Network
- [x] HRNet
- [x] Swin Transformer (Inputs size free)
- [x] MobileNetV2
- [ ] MAE-ViT (Coming soon)
- [x] ConvNeXt

Weights can be downloaded in [here](backbones/weights_download.md)

### Other features
- [x] Mixed precision training and inference
- [x] Fully Deterministic (100%, see https://github.com/NVIDIA/framework-determinism)
- [x] Training and inference on GPU <= 8
- [x] Training and inference on TPU
- [x] Typical image augmentation
- [ ] One click/out of box deployment for industry (Plan mid 2022)

## Requirements

* TensorFlow >= 2.4 (2.3 is fine for TPU users)
* TensorFlow >= 2.6 for Apple M1 users.
* Mixed precision only supports GPU architectures after Volta (included).
