import torch
import dac

class DACAudioTokenizer:
    SAMPLE_RATE = 24000
    FRAME_SIZE = 320  # Matches 24kHz DAC stride

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
        waveform: [1, 1, samples] @ 24kHz mono
        returns:  [1, num_quantizers, time_steps]
        """
        if waveform.dim() != 3 or waveform.shape[1] != 1:
            raise ValueError(f"Expected [B, 1, samples], got {waveform.shape}")

        # print("encode in shape", waveform.shape)
        x = self.model.preprocess(waveform.to(self.device), self.SAMPLE_RATE)
        # print("preprocessed shape", x.shape)
        z, codes, _, _, _ = self.model.encode(x, n_quantizers=self.num_quantizers)
        print(z.shape, codes.shape)
        return codes  # [1, Q, T]
    

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes:    [1, num_quantizers, time_steps]
        returns:  [1, samples] @ 24kHz mono
        """
        
        if codes.dim() != 3 or codes.shape[1] != self.num_quantizers:
            raise ValueError(f"Expected [1, {self.num_quantizers}, T], got {codes.shape}")

        print("codes shape", codes.shape)
        z_out = self.model.quantizer.from_codes(codes.to(self.device))
        z = z_out[0] if isinstance(z_out, tuple) else z_out
        print("zs", z.shape)
        decoded = self.model.decode(z)

        # Guarantee [1, samples] output
        if decoded.dim() == 3:
            decoded = decoded.squeeze(1) if decoded.shape[1] == 1 else decoded.mean(dim=1)
        return decoded