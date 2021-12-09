# iSeg : A TensorFlow 2 Library for Semantic Segmentation

Working in progress

I am very busy at this moment (poor student....), but I will try my best to update this repo until Semantic Segmntation is solved.

I will also upload the code for dataset supports and some implementations of well-known networks (e.g. PSPNet, DeepLab, CCNet, DANet, OCRNet)

Some segmentation results on Flickr images:

<img src="demo.png" width=450>

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
- [x] Training and inference on GPU $\leq$ 8
- [x] Training and inference on TPU
- [x] Typical image augmentation

## Requirements

* TensorFlow >= 2.4 (2.3 is fine for TPU users)
* TensorFlow >= 2.6 for Apple M1 users.
* Mixed precision only supports GPU architectures after Volta (included).
