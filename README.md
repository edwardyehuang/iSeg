# iSeg
## A powerful Tensorflow 2.x based Semantic Segmentation Framework

Working in progress

I am very busy at this moment, but I will try my best to update this repo.

I will also upload the code for dataset supports and some implementations of well-known networks (e.g. PSPNet, DeepLab, CCNet, DANet, OCRNet)

## Features
### Backbone support

- [x] Modern ResNet
- [x] Xception-65
- [x] EfficientNetV1
- [ ] EfficientNetV2
- [ ] Vision Transformer (WIP)
- [ ] MLP Mixer (WIP)
- [ ] SegFormer
- [x] Feature Pyramid Network
- [x] Swin Transformer (Inputs size free)
- [x] MobileNetV2

### Other features
- [x] Mixed precision training and inference
- [x] Deterministic (Currently not 100%, see https://github.com/NVIDIA/framework-determinism)
- [x] Training and inference on TPU
- [x] Typical image augmentation