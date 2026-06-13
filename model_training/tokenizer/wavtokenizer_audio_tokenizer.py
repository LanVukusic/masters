import os
import sys
import torch
from huggingface_hub import hf_hub_download

# Add the cloned WavTokenizer repo to the path so its decoder/encoder packages
# are importable. The repo path is a sibling of the project root.
_WAVTOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "WavTokenizer",
)
if os.path.isdir(_WAVTOKENIZER_PATH) and _WAVTOKENIZER_PATH not in sys.path:
    sys.path.insert(0, _WAVTOKENIZER_PATH)

from decoder.pretrained import WavTokenizer
from model_training.model_config import TARGET_SAMPLING_RATE

HF_REPO = "novateur/WavTokenizer-large-unify-40token"
CKPT_FILE = "wavtokenizer_large_unify_600_24k.ckpt"
WAVTOKENIZER_FRAME_SIZE = 600


class WavTokenizerAudioTokenizer:
    SAMPLE_RATE = TARGET_SAMPLING_RATE
    FRAME_SIZE = WAVTOKENIZER_FRAME_SIZE  # 600 = 24000 / 40 tokens/sec

    def __init__(self, num_quantizers: int = 1, device: str | None = None):
        self.num_quantizers = num_quantizers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        cache_dir = os.path.join(os.path.dirname(__file__), "wavtokenizer_cache")
        os.makedirs(cache_dir, exist_ok=True)

        config_path = os.path.join(
            os.path.dirname(__file__), "wavtokenizer_config.yaml"
        )
        model_path = os.path.join(cache_dir, CKPT_FILE)

        if not os.path.exists(model_path):
            print(f"Downloading WavTokenizer checkpoint from {HF_REPO} ...")
            hf_hub_download(
                repo_id=HF_REPO,
                filename=CKPT_FILE,
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded to {model_path}")

        print(f"Loading WavTokenizer on {self.device}")
        self.model = WavTokenizer.from_pretrained0802(config_path, model_path)
        self.model = self.model.to(self.device).eval()
        self.bandwidth_id = torch.tensor([0], device=self.device)
        print("WavTokenizer ready.")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def frame_size(self) -> int:
        return self.FRAME_SIZE

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [B, 1, samples] @ 24kHz mono
        returns:  [B, num_quantizers, time_steps]
        """
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, samples], got {waveform.shape}")

        wav = waveform.squeeze(1).to(self.device)  # [B, samples]
        _, codes = self.model.encode_infer(wav, bandwidth_id=self.bandwidth_id)
        # codes: [n_q, B, T] -> [B, n_q, T]
        return codes.permute(1, 0, 2)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes:    [B, num_quantizers, time_steps] or [num_quantizers, time_steps]
        returns:  [B, 1, samples] @ 24kHz mono
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

        if codes.dim() != 3 or codes.shape[1] != self.num_quantizers:
            raise ValueError(
                f"Expected [B, {self.num_quantizers}, T], got {codes.shape}"
            )

        codes_wt = codes.permute(1, 0, 2).to(self.device)  # [n_q, B, T]
        features = self.model.codes_to_features(codes_wt)  # [B, C, T]
        audio = self.model.decode(features, bandwidth_id=self.bandwidth_id)  # [B, T]
        return audio.unsqueeze(1)  # [B, 1, T]
