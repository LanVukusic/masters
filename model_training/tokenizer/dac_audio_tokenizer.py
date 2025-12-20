import torch
import torchaudio
import dac
from audiotools import AudioSignal
from typing import Union, Tuple, List
import warnings

model_type = "24khz"  # or "44khz"
SAMPLE_RATE = 24000 if model_type == "24khz" else 44000
SAMPLES_PER_FRAME = 1920


class DACAudioTokenizer:
    """
    A reusable tokenizer class for audio deep learning using DAC (Discrete Audio Codec),
    that handles model loading, CPU/GPU inference, and provides encoding/decoding functionality.
    """

    def __init__(
        self,
        num_quantizers=16,
        device: str = "cpu",
    ):
        """
        Initialize the DACAudioTokenizer.

        Args:
            model_type: DAC model type ("44khz", "24khz", etc.)
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
        """
        self.num_quantizers = num_quantizers
        # self.model_type = model_type
        self.device = self._get_device(device)

        print(f"Loading DAC model {model_type} on {self.device}")

        # Download and load DAC model
        model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(model_path)

        # Move model to specified device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"DAC Model loaded successfully on {self.device}")
        self.sampling_rate = SAMPLE_RATE

    def _get_device(self, device: str = None) -> str:
        """
        Determine the device to run the model on.

        Args:
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)

        Returns:
            The device string ('cpu' or 'cuda')
        """
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return device

    def _prepare_audio(
        self, waveform: torch.Tensor, original_sampling_rate: int
    ) -> torch.Tensor:
        """
        Prepare audio waveform for model input by resampling and normalizing.

        Args:
            waveform: Input audio waveform [channels, samples]
            original_sampling_rate: Original sampling rate of the audio

        Returns:
            Prepared audio tensor ready for model input
        """
        # Resample the audio to the model's required sampling rate
        if original_sampling_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sampling_rate, new_freq=SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def load_audio_from_path(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio from file path.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, original_sampling_rate)
        """
        waveform, original_sampling_rate = torchaudio.load(audio_path)

        num_samples = waveform.shape[1]
        duration = num_samples / original_sampling_rate
        print(
            f"Loaded audio: {num_samples} samples, {original_sampling_rate}Hz, {duration:.2f}s"
        )

        return waveform, original_sampling_rate

    def encode_from_path(
        self, audio_path: Union[str, List[str]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Encode audio from file path(s) to DAC codes format.

        Args:
            audio_path: Path to audio file or list of audio file paths

        Returns:
            Encoded audio tokens [batch, quantizers, time_steps] for single path,
            or list of encoded tensors for multiple paths
        """
        if isinstance(audio_path, str):
            # Single audio file
            waveform, original_sampling_rate = self.load_audio_from_path(audio_path)
            return self._encode(waveform, original_sampling_rate)
        elif isinstance(audio_path, list):
            # Multiple audio files
            encoded_batch = [self.encode_from_path(path) for path in audio_path]
            return encoded_batch
        else:
            raise ValueError("audio_path must be a string or list of strings")

    def encode_from_waveform(
        self,
        waveform: Union[torch.Tensor, List[torch.Tensor]],
        original_sampling_rate: int,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Encode raw audio waveform(s) to DAC codes format.

        Args:
            waveform: Raw audio waveform [channels, samples] or list of waveforms
            original_sampling_rate: Original sampling rate of the audio

        Returns:
            Encoded audio tokens [batch, quantizers, time_steps] for single waveform,
            or list of encoded tensors for multiple waveforms
        """
        if isinstance(waveform, torch.Tensor):
            # Single waveform
            return self._encode(waveform, original_sampling_rate)
        elif isinstance(waveform, list):
            # Multiple waveforms
            encoded_batch = [
                self.encode_from_waveform(single_waveform, original_sampling_rate)
                for single_waveform in waveform
            ]
            return encoded_batch
        else:
            raise ValueError("waveform must be a torch.Tensor or list of torch.Tensors")

    def _encode(
        self,
        waveform: torch.Tensor,
        original_sampling_rate: int,
    ) -> torch.Tensor:
        """
        Core encoding logic using DAC.

        Args:
            waveform: Raw audio waveform [channels, samples]
            original_sampling_rate: Original sampling rate of the audio

        Returns:
            Encoded audio tokens [batch, quantizers, time_steps] similar to Mimi format
        """
        # Prepare the audio
        prepared_waveform = self._prepare_audio(waveform, original_sampling_rate)
        # print("prettparing...")

        # Convert torch tensor to numpy array
        audio_np = prepared_waveform.squeeze().cpu().numpy()

        # Create AudioSignal object
        signal = AudioSignal(audio_np, sample_rate=SAMPLE_RATE)

        # Move signal to model's device
        signal.to(self.model.device)

        # Preprocess and encode to get the actual codes
        x = self.model.preprocess(signal.audio_data, signal.sample_rate)
        z, codes, latents, _, _ = self.model.encode(x, n_quantizers=self.num_quantizers)

        # codes   # [batch, n_quantizers, time_steps]
        # print("prepared")
        return codes

    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode DAC codes back to raw waveform.

        Args:
            codes: Encoded audio tokens [batch, quantizers, time_steps]

        Returns:
            Decoded audio waveform [batch, channels, samples]
        """
        # Convert codes back to DAC format [n_quantizers, batch, time_steps]
        codes_dac_format = codes.permute(1, 0, 2)  # [quantizers, batch, time_steps]

        # Decode using DAC model - need to handle the quantizer output properly
        with torch.no_grad():
            # The quantizer.from_codes returns a tuple, we need the first element
            # which is the actual latent representation
            z_tuple = self.model.quantizer.from_codes(codes_dac_format)
            if isinstance(z_tuple, tuple):
                z = z_tuple[0]  # Take the first element of the tuple
            else:
                z = z_tuple  # If it's not a tuple, use as is

            # The decode method expects the latent representation and returns [batch, channels, samples]
            # But it returns with batch and channels swapped, so we need to fix the shape
            decoded_audio = self.model.decode(z)

        # The DAC decode returns [batch, channels, samples] but we might need to ensure correct shape
        # If the shape is [n_quantizers, channels, samples] due to the batch dimension being wrong,
        # we need to handle it properly
        if decoded_audio.shape[0] != codes.shape[0]:  # batch dimension doesn't match
            # The decoded audio should have the same batch size as input codes
            # DAC decode might return [batch, channels, samples] correctly
            pass  # The shape should be correct as [batch, channels, samples]

        return decoded_audio.to(self.device)

    def decode_to_waveform(
        self, codes: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Decode DAC codes back to raw waveform(s).

        Args:
            codes: Encoded audio tokens [batch, quantizers, time_steps] or list of tensors

        Returns:
            Decoded audio waveform [batch, channels, samples] for single input,
            or list of decoded waveforms for multiple inputs
        """
        if isinstance(codes, torch.Tensor):
            # Single codes tensor
            return self.decode_from_codes(codes)
        elif isinstance(codes, list):
            # Multiple codes tensors
            decoded_batch = []
            for single_codes in codes:
                decoded = self.decode_from_codes(single_codes)
                decoded_batch.append(decoded)
            return decoded_batch
        else:
            raise ValueError("codes must be a torch.Tensor or list of torch.Tensors")

    def decode_to_file(
        self,
        codes: Union[torch.Tensor, List[torch.Tensor]],
        output_path: Union[str, List[str]],
        format: str = "wav",
    ):
        """
        Decode DAC codes and save to file(s).

        Args:
            codes: Encoded audio tokens [batch, quantizers, time_steps] or list of tensors
            output_path: Path to save the output audio file or list of paths
            format: Audio format to save as
        """
        if isinstance(codes, list):
            if not isinstance(output_path, list):
                # Multiple codes and single output path (save as batch with numbered files)
                base_path = output_path.rsplit(".", 1)[0]
                ext = output_path.rsplit(".", 1)[1] if "." in output_path else "wav"
                output_path = [f"{base_path}_{i}.{ext}" for i in range(len(codes))]

            if len(codes) != len(output_path):
                raise ValueError("Number of codes must match number of output paths")

            for single_codes, path in zip(codes, output_path):
                self.decode_to_file(single_codes, path, format)
        elif isinstance(codes, torch.Tensor):
            if not isinstance(output_path, str):
                raise ValueError("output_path must be a string")

            # Decode the codes to waveform
            decoded_waveform = self.decode_from_codes(codes)

            # Convert to AudioSignal for saving
            # Remove batch dimension if present for single audio
            if decoded_waveform.dim() == 3 and decoded_waveform.shape[0] == 1:
                audio_data = decoded_waveform.squeeze(0)
            else:
                audio_data = decoded_waveform

            # Create AudioSignal and save
            signal = AudioSignal(audio_data.cpu().numpy(), sample_rate=SAMPLE_RATE)
            signal.write(output_path)
            print(f"Audio saved to {output_path}")
        else:
            raise ValueError("codes must be a torch.Tensor or list of torch.Tensors")

    def encode_decode_cycle(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a complete encode-decode cycle.

        Args:
            audio_path: Path to input audio file

        Returns:
            Tuple of (original_waveform, decoded_waveform)
        """
        # Load original audio
        original_waveform, original_sr = self.load_audio_from_path(audio_path)
        prepared_original = self._prepare_audio(original_waveform, original_sr)

        # Encode
        compressed_obj = self.encode_from_path(audio_path)

        # Decode
        decoded_waveform = self.decode_to_waveform(compressed_obj)

        # Remove extra batch dimension if present to match original shape
        if decoded_waveform.dim() == 3 and prepared_original.dim() == 2:
            decoded_waveform = decoded_waveform.squeeze(0)

        return prepared_original, decoded_waveform

    def batch_encode(self, audio_paths: list) -> list:
        """
        Encode multiple audio files in batch.

        Args:
            audio_paths: List of audio file paths

        Returns:
            List of compressed DAC objects
        """
        encoded_batch = []
        for path in audio_paths:
            encoded = self.encode_from_path(path)
            encoded_batch.append(encoded)
        return encoded_batch

    def set_device(self, device: str):
        """
        Change the device for inference.

        Args:
            device: New device ('cpu' or 'cuda')
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError("Device must be 'cpu' or 'cuda'")

        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            device = "cpu"

        self.device = device
        self.model = self.model.to(device)
        print(f"DAC Model moved to {device}")

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model_type,
            "sampling_rate": SAMPLE_RATE,
            "device": self.device,
            "is_cuda_available": torch.cuda.is_available(),
            "model_on_gpu": next(self.model.parameters()).is_cuda
            if hasattr(next(self.model.parameters()), "is_cuda")
            else False,
        }
