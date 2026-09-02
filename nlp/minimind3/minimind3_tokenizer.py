from tokenizers import Tokenizer


class MiniMind3Tokenizer:
    """Tokenizer wrapper around the standalone `tokenizers` library.

    Loads the official tokenizer.json shipped with minimind-3 and reproduces
    the chat prompt format of chat_template.jinja (single/multi-turn, no
    tool calls, thinking disabled).
    """

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def __init__(self, tokenizer_path):
        self._tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text):
        return self._tokenizer.encode(text).ids

    def decode(self, token_ids, skip_special_tokens=True):
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def build_chat_prompt(self, messages, add_generation_prompt=True):
        """Formats a list of {"role": ..., "content": ...} messages.

        Matches chat_template.jinja with `open_thinking=False`: the
        generation prompt ends with an empty `<think></think>` block.
        """
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"{self.IM_START}system\n{content}{self.IM_END}\n"
            elif role == "user":
                prompt += f"{self.IM_START}user\n{content}{self.IM_END}\n"
            elif role == "assistant":
                prompt += (
                    f"{self.IM_START}assistant\n<think>\n\n</think>\n\n"
                    f"{content.lstrip()}{self.IM_END}\n"
                )
            else:
                raise ValueError(f"Unsupported role: {role}")
        if add_generation_prompt:
            prompt += f"{self.IM_START}assistant\n<think>\n\n</think>\n\n"
        return prompt
