class MiniMind3Config:
    """Hyperparameters of the minimind-3 dense model (Qwen3-style architecture).

    Mirrors /share/edwardyehuang/huggingface/minimind-3/config.json.
    """

    def __init__(
        self,
        vocab_size=6400,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=96,
        intermediate_size=2432,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
