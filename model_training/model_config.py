# DAC produces ~75 tokens/second at 24kHz (not 100!)
TARGET_SAMPLING_RATE = 24000
DAC_FRAME_SIZE = 320  # samples per DAC token at 24kHz
WAVTOKENIZER_FRAME_SIZE = 600  # samples per WavTokenizer token at 24kHz (40 tps)
TOKEN_RATE = TARGET_SAMPLING_RATE / DAC_FRAME_SIZE
DAC_SAMPLES_PER_TOKEN = DAC_FRAME_SIZE


def tokens_to_samples(tokens: int, frame_size: int = DAC_FRAME_SIZE) -> int:
    return tokens * frame_size


def samples_to_tokens(samples: int, frame_size: int = DAC_FRAME_SIZE) -> int:
    return samples // frame_size


def seconds_to_samples(seconds: float, sample_rate: int = TARGET_SAMPLING_RATE) -> int:
    return int(seconds * sample_rate)


def seconds_to_tokens(
    seconds: float,
    sample_rate: int = TARGET_SAMPLING_RATE,
    frame_size: int = DAC_FRAME_SIZE,
) -> int:
    return int(round(seconds * sample_rate / frame_size))


def samples_to_chunks(samples: int, frame_size: int = DAC_FRAME_SIZE) -> int:
    return (samples + frame_size - 1) // frame_size


def tokens_to_chunks(tokens: int, frame_size: int = DAC_FRAME_SIZE) -> int:
    return samples_to_chunks(tokens_to_samples(tokens, frame_size), frame_size)


MODEL_CONFIG = {
    "target_sampling_rate": TARGET_SAMPLING_RATE,
    "past_len": int(8 * TOKEN_RATE),
    "future_len": int(2 * TOKEN_RATE),
    # "future_len": int(3 * TOKEN_RATE),
    "vocab_size": 1024,
    "n_codebooks": 4,
    "d_model": 1024,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 1024,  # 4 * d_model — standard transformer ratio
    "dropout": 0.1,
}
