import torch
from encodec import EncodecModel
from model_training.model_config import ENCODEC_FRAME_SIZE, TARGET_SAMPLING_RATE

# EnCodec 24kHz bandwidth → quantizers mapping
_ENCODEC_BANDWIDTHS = {2: 1.5, 4: 3.0, 8: 6.0, 16: 12.0, 32: 24.0}


class EnCodecAudioTokenizer:
    SAMPLE_RATE = TARGET_SAMPLING_RATE
    FRAME_SIZE = ENCODEC_FRAME_SIZE

    def __init__(self, num_quantizers: int = 4, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        valid = sorted(_ENCODEC_BANDWIDTHS.keys())
        clamped = min(valid, key=lambda x: abs(x - num_quantizers))
        if clamped != num_quantizers:
            print(
                f"EnCodec: clamping num_quantizers={num_quantizers} -> {clamped} "
                f"(valid: {valid})"
            )
        self.num_quantizers = clamped
        bandwidth = _ENCODEC_BANDWIDTHS[clamped]

        print(
            f"Loading EnCodec 24kHz model on {self.device} "
            f"(bandwidth={bandwidth} kbps, {self.num_quantizers} codebooks)"
        )
        self.model = EncodecModel.encodec_model_24khz()
        self.model = self.model.to(self.device).eval()
        self.model.set_target_bandwidth(bandwidth)
        print("EnCodec ready.")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def frame_size(self) -> int:
        return self.FRAME_SIZE

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [B, 1, samples] @ 24kHz mono
        returns:  [B, num_quantizers, time_steps]
        """
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, samples], got {waveform.shape}")

        encoded_frames = self.model.encode(waveform.to(self.device))
        return encoded_frames[0][0]  # [B, K, T]

    @torch.no_grad()
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

        decoded = self.model.decode([(codes.to(self.device), None)])
        if decoded.shape[1] == 1:
            return decoded
        return decoded.mean(dim=1, keepdim=True)
