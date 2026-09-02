import tensorflow as tf


def _apply_top_k(logits, top_k):
    if top_k is None or top_k <= 0:
        return logits
    kth_values = tf.math.top_k(logits, top_k).values[:, -1:]
    return tf.where(
        logits < kth_values, tf.cast(-float("inf"), logits.dtype), logits
    )


def _apply_top_p(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits = tf.sort(logits, direction="DESCENDING", axis=-1)
    sorted_indices = tf.argsort(logits, direction="DESCENDING", axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_mask = cumulative_probs > top_p
    # Shift right so the first token above the threshold is kept.
    sorted_mask = tf.concat(
        [tf.zeros_like(sorted_mask[:, :1], dtype=tf.bool), sorted_mask[:, :-1]],
        axis=-1,
    )
    sorted_logits = tf.where(
        sorted_mask, tf.cast(-float("inf"), logits.dtype), sorted_logits
    )
    inverse_indices = tf.argsort(sorted_indices, axis=-1)
    return tf.gather(sorted_logits, inverse_indices, batch_dims=1)


def sample_next_token(logits, temperature=0.85, top_p=0.95, top_k=50):
    """Samples one token from logits [batch, vocab], matching the sampling
    pipeline of minimind's `generate` (temperature -> top_k -> top_p)."""
    logits = tf.cast(logits, tf.float32) / temperature
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)
    return tf.random.categorical(logits, num_samples=1)


def generate(
    model,
    input_ids,
    max_new_tokens=512,
    temperature=0.85,
    top_p=0.95,
    top_k=50,
    eos_token_id=2,
    seed=None,
):
    """Naive sampling loop without KV cache (recomputes the full prefix).

    Args:
        model: `MiniMind3Model` returning logits [batch, seq_len, vocab].
        input_ids: list of ints (single sequence, batch size 1).
        max_new_tokens: maximum number of tokens to generate.
        temperature / top_p / top_k: sampling hyperparameters.
        eos_token_id: stop token id.
        seed: optional int for reproducible sampling.

    Returns:
        List of generated token ids (excluding the prompt).
    """
    if seed is not None:
        tf.random.set_seed(seed)
    ids = tf.constant([input_ids], dtype=tf.int32)
    generated = []
    for _ in range(max_new_tokens):
        logits = model(ids, training=False)[:, -1, :]
        next_token = sample_next_token(
            logits, temperature=temperature, top_p=top_p, top_k=top_k
        )
        next_id = int(next_token[0, 0].numpy())
        generated.append(next_id)
        ids = tf.concat(
            [ids, tf.constant([[next_id]], dtype=tf.int32)], axis=1
        )
        if eos_token_id is not None and next_id == eos_token_id:
            break
    return generated
