import torch
import dac
from model_training.model_config import DAC_FRAME_SIZE, TARGET_SAMPLING_RATE

class DACAudioTokenizer:
    SAMPLE_RATE = TARGET_SAMPLING_RATE
    FRAME_SIZE = DAC_FRAME_SIZE  # Matches 24kHz DAC stride

    def __init__(self, num_quantizers: int = 9, device: str | None = None):
        self.num_quantizers = num_quantizers
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading DAC 24kHz model on {self.device}")
        model_path = dac.utils.download(model_type="24khz")
        self.model = dac.DAC.load(model_path).to(self.device).eval()
        print("Model ready.")

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def frame_size(self) -> int:
        """
        Frame size represents how many raw waveform samples are needed to generate one dac frame or token
        """
        return self.FRAME_SIZE
    
    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: [B, 1, samples] @ 24kHz mono
        returns:  [B, num_quantizers, time_steps]
        """
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, samples], got {waveform.shape}")

        x = self.model.preprocess(waveform.to(self.device), self.SAMPLE_RATE)
        z, codes, _, _, _ = self.model.encode(x, n_quantizers=self.num_quantizers)
        return codes  # [1, Q, T]
    

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes:    [B, num_quantizers, time_steps] or [num_quantizers, time_steps]
        returns: [B, 1, samples] @ 24kHz mono
        """
        
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

        if codes.dim() != 3 or codes.shape[1] != self.num_quantizers:
            raise ValueError(
                f"Expected [B, {self.num_quantizers}, T], got {codes.shape}"
            )

        z_out = self.model.quantizer.from_codes(codes.to(self.device))
        z = z_out[0] if isinstance(z_out, tuple) else z_out
        decoded = self.model.decode(z)

        if decoded.dim() == 3:
            if decoded.shape[1] == 1:
                decoded = decoded
            else:
                decoded = decoded.mean(dim=1, keepdim=True)
        elif decoded.dim() == 2:
            decoded = decoded.unsqueeze(1)
        return decoded