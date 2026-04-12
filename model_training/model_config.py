MODEL_CONFIG = {
    "target_sampling_rate": 24000,
    "past_len": int(2 * 100),
    "future_len": int(3 * 100),
    "vocab_size": 1024,
    "n_codebooks": 9,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "d_ff": 128,
    "dropout": 0.1,
}

TOKEN_RATE = 100
