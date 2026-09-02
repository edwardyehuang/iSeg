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


def _get_compiled_fns(model):
    """Returns cached (prefill_fn, decode_fn) tf.functions for `model`.

    Graph compilation removes the per-op eager overhead that otherwise
    dominates single-token decode steps. Dynamic sequence dimensions keep
    each function to a single trace.
    """
    fns = getattr(model, "_minimind3_compiled_fns", None)
    if fns is not None:
        return fns

    num_layers = len(model.blocks)
    cache_spec = tf.TensorSpec(
        [1, None, model.config.num_key_value_heads, model.config.head_dim],
        tf.float32,
    )
    caches_spec = [(cache_spec, cache_spec) for _ in range(num_layers)]

    @tf.function(input_signature=[tf.TensorSpec([1, None], tf.int32)])
    def prefill_fn(input_ids):
        return model(input_ids, caches=None, start_pos=0)

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, 1], tf.int32),
            caches_spec,
            tf.TensorSpec([], tf.int32),
        ]
    )
    def decode_fn(token_id, caches, start_pos):
        return model(token_id, caches=caches, start_pos=start_pos)

    fns = (prefill_fn, decode_fn)
    model._minimind3_compiled_fns = fns
    return fns


def generate(
    model,
    input_ids,
    max_new_tokens=512,
    temperature=0.85,
    top_p=0.95,
    top_k=50,
    eos_token_id=2,
    seed=None,
    use_cache=True,
):
    """Sampling loop with optional KV cache.

    Args:
        model: `MiniMind3Model` returning (logits, caches).
        input_ids: list of ints (single sequence, batch size 1).
        max_new_tokens: maximum number of tokens to generate.
        temperature / top_p / top_k: sampling hyperparameters.
        eos_token_id: stop token id.
        seed: optional int for reproducible sampling.
        use_cache: if True, only the newly generated token is fed at each
            step (KV cache, graph compiled); if False, the full prefix is
            recomputed eagerly each step.

    Returns:
        List of generated token ids (excluding the prompt).
    """
    if seed is not None:
        tf.random.set_seed(seed)
    prompt_len = len(input_ids)

    if use_cache:
        prefill_fn, decode_fn = _get_compiled_fns(model)
        logits, caches = prefill_fn(tf.constant([input_ids], dtype=tf.int32))
        generated = []
        for step in range(max_new_tokens):
            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            next_id = int(next_token[0, 0].numpy())
            generated.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                break
            if step + 1 < max_new_tokens:
                logits, caches = decode_fn(
                    tf.constant([[next_id]], dtype=tf.int32),
                    caches,
                    tf.constant(prompt_len + step, dtype=tf.int32),
                )
        return generated

    # No-cache path: recompute the full prefix each step (eager).
    ids = tf.constant([input_ids], dtype=tf.int32)
    generated = []
    for _ in range(max_new_tokens):
        logits, _ = model(ids, caches=None, start_pos=0)
        next_token = sample_next_token(
            logits[:, -1, :], temperature=temperature, top_p=top_p, top_k=top_k
        )
        next_id = int(next_token[0, 0].numpy())
        generated.append(next_id)
        ids = tf.concat(
            [ids, tf.constant([[next_id]], dtype=tf.int32)], axis=1
        )
        if eos_token_id is not None and next_id == eos_token_id:
            break
    return generated
