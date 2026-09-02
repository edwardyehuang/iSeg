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
    `call(input_ids)` returns logits of shape [batch, seq_len, vocab_size].
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

    def call(self, input_ids):
        x = self.embed_tokens(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Tied lm head: logits = hidden @ embedding^T.
        logits = tf.matmul(x, self.embed_tokens.embeddings, transpose_b=True)
        return logits

    def build_model(self):
        """Builds all weights eagerly (input shape [batch, seq_len])."""
        self(tf.zeros((1, 1), dtype=tf.int32))
        return self
