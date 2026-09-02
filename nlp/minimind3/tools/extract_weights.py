"""Extracts minimind-3 torch/safetensors weights into a plain numpy .npz.

Run this in the `minimind` conda environment (has torch + safetensors):

    python iseg/nlp/minimind3/tools/extract_weights.py

The resulting .npz keeps the original safetensors key names and can be
loaded in the tf215 environment by `convert_weights.py`.
"""

import argparse

import numpy as np
from safetensors.numpy import load_file

DEFAULT_SAFETENSORS = (
    "/share/edwardyehuang/huggingface/minimind-3/model.safetensors"
)
DEFAULT_OUTPUT = "/home/edwardyehuang/weights/minimind3_numpy.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safetensors", default=DEFAULT_SAFETENSORS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state_dict = load_file(args.safetensors)
    print(f"Loaded {len(state_dict)} tensors from {args.safetensors}")
    for key, value in sorted(state_dict.items()):
        print(f"  {key}: {value.shape} {value.dtype}")

    np.savez(args.output, **state_dict)
    print(f"Saved numpy weights to {args.output}")


if __name__ == "__main__":
    main()
