"""Simple chat demo / sanity test for the ported minimind-3 Keras model.

Run in the `tf215` conda environment from the segmenty repo root:

    PYTHONPATH=. python iseg/nlp/minimind3/chat_demo.py
"""

import argparse
import time

import tensorflow as tf

from iseg.nlp.minimind3.minimind3_config import MiniMind3Config
from iseg.nlp.minimind3.minimind3_model import MiniMind3Model
from iseg.nlp.minimind3.minimind3_tokenizer import MiniMind3Tokenizer
from iseg.nlp.minimind3.minimind3_generate import generate

DEFAULT_WEIGHTS = "/home/edwardyehuang/weights/minimind3.h5"
DEFAULT_TOKENIZER = (
    "/share/edwardyehuang/huggingface/minimind-3/tokenizer.json"
)

DEFAULT_QUESTIONS = [
    "介绍一下福建省",
    "中国的首都是哪里？",
    "为什么天空是蓝色的？",
    "推荐一些中国的美食",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = MiniMind3Config()
    model = MiniMind3Model(config)
    model.build_model()
    model.load_weights(args.weights)
    print(f"Loaded weights from {args.weights}")

    tokenizer = MiniMind3Tokenizer(args.tokenizer)

    for question in DEFAULT_QUESTIONS:
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.build_chat_prompt(messages)
        input_ids = tokenizer.encode(prompt)

        start_time = time.time()
        output_ids = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_token_id=config.eos_token_id,
            seed=args.seed,
        )
        elapsed = time.time() - start_time

        response = tokenizer.decode(output_ids)
        print("=" * 60)
        print(f"Q: {question}")
        print(f"A: {response}")
        print(f"[{len(output_ids)} tokens in {elapsed:.1f}s]")


if __name__ == "__main__":
    main()
