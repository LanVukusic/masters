from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvaluationConfig:
    checkpoint_path: Optional[str] = None
    audio_path: str = "dataset_gen/free_music/mp3s/"
    audio_file: str = "100066 Lindstrom - Monsteer (Original Mix).mp3"

    step_seconds: float = 0.5
    predict_seconds: float = 1.0
    past_seconds: float = 2.0
    max_windows: Optional[int] = None

    output_dir: str = "research/continuation/results"

    temperature: float = 1.0
    top_k: Optional[int] = 200
    top_p: float = 0.95
    repetition_penalty: float = 1.1

    batch_size: int = 1

    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"


DEFAULT_CONFIG = EvaluationConfig()
