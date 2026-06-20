TARGET_SAMPLING_RATE = 24000
DAC_FRAME_SIZE = 320  # samples per DAC token at 24kHz (75 tps)
WAVTOKENIZER_FRAME_SIZE = 600  # samples per WavTokenizer token at 24kHz (40 tps)
ENCODEC_FRAME_SIZE = (
    320  # samples per EnCodec token at 24kHz (75 tps, same stride as DAC)
)

# Desired audio duration in seconds — stable across tokenizer choices.
PAST_DURATION_SECONDS = 10.0
FUTURE_DURATION_SECONDS = 3.0


def compute_token_lengths(frame_size: int) -> tuple[int, int]:
    token_rate = TARGET_SAMPLING_RATE / frame_size
    past_len = int(PAST_DURATION_SECONDS * token_rate)
    future_len = int(FUTURE_DURATION_SECONDS * token_rate)
    return past_len, future_len


def tokens_to_samples(tokens: int, frame_size: int) -> int:
    return tokens * frame_size


def samples_to_tokens(samples: int, frame_size: int) -> int:
    return samples // frame_size


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return int(seconds * sample_rate)


def seconds_to_tokens(
    seconds: float,
    frame_size: int,
    sample_rate: int,
) -> int:
    return int(round(seconds * sample_rate / frame_size))


def samples_to_chunks(samples: int, frame_size: int) -> int:
    return (samples + frame_size - 1) // frame_size


def tokens_to_chunks(tokens: int, frame_size: int) -> int:
    return samples_to_chunks(tokens_to_samples(tokens, frame_size), frame_size)


# MODEL_CONFIG = {
#     "target_sampling_rate": TARGET_SAMPLING_RATE,
#     "past_len": int(6 * TOKEN_RATE),
#     "future_len": int(2 * TOKEN_RATE),
#     # "future_len": int(3 * TOKEN_RATE),
#     "vocab_size": 1024,
#     "n_codebooks": 4,
#     "d_model": 512,
#     "n_heads": 8,
#     "n_layers": 6,
#     "d_ff": 1024,  # 4 * d_model — standard transformer ratio
#     "dropout": 0.1,
# }

MODEL_CONFIG = {
    "n_codebooks": 4,  # Try bumping from 4 to 8 if the 4-layer output is too noisy
    "vocab_size": 1024,
    "d_model": 768,  # Up from 512. More capacity per token.
    "n_heads": 12,  # Standard ratio for d_model=768 (768/12=64)
    "n_layers": 8,  # Up from 6. Crucial for temporal depth.
    "d_ff": 1024,  # 3x or 4x d_model. Let's give it room to think.
    "dropout": 0.1,
}
