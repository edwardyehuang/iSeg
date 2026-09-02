"""Converts the extracted numpy weights into the Keras minimind3 model and
saves them with `model.save_weights` (.h5).

Run this in the `tf215` conda environment:

    python iseg/nlp/minimind3/tools/convert_weights.py
"""

import argparse

import numpy as np
import tensorflow as tf

from iseg.nlp.minimind3.minimind3_config import MiniMind3Config
from iseg.nlp.minimind3.minimind3_model import MiniMind3Model

DEFAULT_INPUT = "/home/edwardyehuang/weights/minimind3_numpy.npz"
DEFAULT_OUTPUT = "/home/edwardyehuang/weights/minimind3.h5"


def assign(variable, value, transpose=False, name=""):
    value = np.asarray(value, dtype=np.float32)
    if transpose:
        # torch Linear stores [out_features, in_features];
        # keras Dense kernel stores [in_features, out_features].
        value = value.T
    if tuple(variable.shape) != tuple(value.shape):
        raise ValueError(
            f"Shape mismatch for {name}: "
            f"variable {tuple(variable.shape)} vs value {value.shape}"
        )
    variable.assign(value)


def convert(npz_path, output_path):
    weights = dict(np.load(npz_path))
    print(f"Loaded {len(weights)} numpy tensors from {npz_path}")

    config = MiniMind3Config()
    model = MiniMind3Model(config)
    model.build_model()

    consumed = set()

    def take(key):
        consumed.add(key)
        return weights[key]

    assign(
        model.embed_tokens.embeddings,
        take("model.embed_tokens.weight"),
        name="embed_tokens",
    )

    for i, block in enumerate(model.blocks):
        prefix = f"model.layers.{i}"
        attn = block.self_attn
        assign(block.input_layernorm.scale_weight, take(f"{prefix}.input_layernorm.weight"), name=f"{prefix}.input_layernorm")
        assign(attn.q_proj.kernel, take(f"{prefix}.self_attn.q_proj.weight"), transpose=True, name=f"{prefix}.q_proj")
        assign(attn.k_proj.kernel, take(f"{prefix}.self_attn.k_proj.weight"), transpose=True, name=f"{prefix}.k_proj")
        assign(attn.v_proj.kernel, take(f"{prefix}.self_attn.v_proj.weight"), transpose=True, name=f"{prefix}.v_proj")
        assign(attn.o_proj.kernel, take(f"{prefix}.self_attn.o_proj.weight"), transpose=True, name=f"{prefix}.o_proj")
        assign(attn.q_norm.scale_weight, take(f"{prefix}.self_attn.q_norm.weight"), name=f"{prefix}.q_norm")
        assign(attn.k_norm.scale_weight, take(f"{prefix}.self_attn.k_norm.weight"), name=f"{prefix}.k_norm")
        assign(block.post_attention_layernorm.scale_weight, take(f"{prefix}.post_attention_layernorm.weight"), name=f"{prefix}.post_attention_layernorm")
        assign(block.mlp.gate_proj.kernel, take(f"{prefix}.mlp.gate_proj.weight"), transpose=True, name=f"{prefix}.gate_proj")
        assign(block.mlp.up_proj.kernel, take(f"{prefix}.mlp.up_proj.weight"), transpose=True, name=f"{prefix}.up_proj")
        assign(block.mlp.down_proj.kernel, take(f"{prefix}.mlp.down_proj.weight"), transpose=True, name=f"{prefix}.down_proj")

    assign(model.norm.scale_weight, take("model.norm.weight"), name="model.norm")

    leftover = set(weights.keys()) - consumed
    # lm_head is tied to the embedding in the original checkpoint, so it is
    # expected to be absent from the safetensors file.
    assert not leftover, f"Unconverted weights left in npz: {leftover}"

    # Sanity check: forward pass on a dummy token.
    logits, _ = model(tf.constant([[1, 2]], dtype=tf.int32))
    print(f"Sanity forward: logits shape {logits.shape}, "
          f"mean {float(tf.reduce_mean(logits)):.4f}, "
          f"std {float(tf.math.reduce_std(logits)):.4f}")

    model.save_weights(output_path)
    print(f"Saved Keras weights to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
