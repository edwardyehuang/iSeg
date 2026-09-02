import tensorflow as tf
import keras

from iseg.nlp.minimind3.minimind3_config import MiniMind3Config
from iseg.nlp.minimind3.minimind3_layers import (
    MiniMind3RMSNorm,
    MiniMind3DecoderBlock,
)


class MiniMind3Model(keras.Model):
    """minimind-3 decoder-only language model (dense, Qwen3-style).

    The lm head is tied to the token embedding, as in the original model.
    `call(input_ids)` returns `(logits, caches)`, where logits have shape
    [batch, seq_len, vocab_size].
    """

    def __init__(self, config: MiniMind3Config = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or MiniMind3Config()

        self.embed_tokens = keras.layers.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            name="embed_tokens",
        )
        self.blocks = [
            MiniMind3DecoderBlock(self.config, name=f"layer_{i}")
            for i in range(self.config.num_hidden_layers)
        ]
        self.norm = MiniMind3RMSNorm(
            epsilon=self.config.rms_norm_eps, name="norm"
        )

    def call(self, input_ids, caches=None, start_pos=0):
        """Runs the model.

        Args:
            input_ids: int tensor [batch, seq_len].
            caches: optional list of per-layer (key, value) cache tuples.
            start_pos: int. Position offset of the first input token.

        Returns:
            (logits, new_caches): logits [batch, seq_len, vocab_size] and
            the updated per-layer caches.
        """
        x = self.embed_tokens(input_ids)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x, new_cache = block(x, cache=cache, start_pos=start_pos)
            new_caches.append(new_cache)
        x = self.norm(x)
        # Tied lm head: logits = hidden @ embedding^T.
        logits = tf.matmul(x, self.embed_tokens.embeddings, transpose_b=True)
        return logits, new_caches

    def build_model(self):
        """Builds all weights eagerly (input shape [batch, seq_len])."""
        self(tf.zeros((1, 1), dtype=tf.int32))
        return self
