# DAC produces ~75 tokens/second at 24kHz (not 100!)
TOKEN_RATE = 75
DAC_SAMPLES_PER_TOKEN = 24000 / TOKEN_RATE  # ~320 samples per token

MODEL_CONFIG = {
    "target_sampling_rate": 24000,
    "past_len": int(2 * TOKEN_RATE),
    "future_len": int(3 * TOKEN_RATE),
    # "future_len": int(3 * TOKEN_RATE),
    "vocab_size": 1024,
    "n_codebooks": 12,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 3,
    "d_ff": 128,
    "dropout": 0.1,
}
