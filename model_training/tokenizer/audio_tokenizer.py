import torch
import torchaudio
import numpy as np
from transformers import MimiModel, AutoFeatureExtractor
from typing import Union, Optional, Tuple, List
import warnings

SAMPLES_PER_FRAME = 1920
SAMPLE_RATE = 24000  # 2.4KHZ


class AudioTokenizer:
    """
    A reusable tokenizer class for audio deep learning that handles model loading,
    CPU/GPU inference, and provides encoding/decoding functionality.
    """

    def __init__(
        self,
        num_quantizers: int = 32,
        device: str = None,
        model_string: str = "kyutai/mimi",
    ):
        """
        Initialize the AudioTokenizer.

        Args:
            model_string: Hugging Face model identifier
            num_quantizers: Number of quantizers to use (exposed as variable)
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_string = model_string
        self.num_quantizers = num_quantizers
        self.device = self._get_device(device)

        print(f"Loading model {model_string} from_pretrained on {self.device}")

        # Load model and feature extractor
        self.model = MimiModel.from_pretrained(model_string)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_string)

        # Move model to specified device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"Model loaded successfully on {self.device}")

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

    @property
    def sampling_rate(self) -> int:
        """Get the model's required sampling rate."""
        return self.feature_extractor.sampling_rate

    @property
    def frame_size(self) -> int:
        """Get the model's frame size for streaming."""
        # This is an approximation - you may need to adjust based on the actual model
        return 1920  # Common frame size for audio models

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
        if original_sampling_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sampling_rate, new_freq=self.sampling_rate
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
        self, audio_path: Union[str, List[str]], num_quantizers: Optional[int] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Encode audio from file path(s) to tokens.

        Args:
            audio_path: Path to audio file or list of audio file paths
            num_quantizers: Number of quantizers to use (uses default if None)

        Returns:
            Encoded audio tokens [batch, quantizers, time_steps] for single path,
            or list of encoded tensors for multiple paths
        """
        if isinstance(audio_path, str):
            # Single audio file
            waveform, original_sampling_rate = self.load_audio_from_path(audio_path)
            return self._encode(waveform, original_sampling_rate, num_quantizers)
        elif isinstance(audio_path, list):
            # Multiple audio files
            encoded_batch = [
                self.encode_from_path(path, num_quantizers) for path in audio_path
            ]
            return encoded_batch
        else:
            raise ValueError("audio_path must be a string or list of strings")

    def encode_from_waveform(
        self,
        waveform: Union[torch.Tensor, List[torch.Tensor]],
        original_sampling_rate: int,
        num_quantizers: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Encode raw audio waveform(s) tokens.

        Args:
            waveform: Raw audio waveform [channels, samples] or list of waveforms
            original_sampling_rate: Original sampling rate of the audio
            num_quantizers: Number of quantizers to use (uses default if None)

        Returns:
            Encoded audio tokens [batch, quantizers, time_steps] for single waveform,
            or list of encoded tensors for multiple waveforms
        """
        if isinstance(waveform, torch.Tensor):
            # Single waveform
            return self._encode(waveform, original_sampling_rate, num_quantizers)
        elif isinstance(waveform, list):
            # Multiple waveforms
            encoded_batch = [
                self.encode_from_waveform(
                    single_waveform, original_sampling_rate, num_quantizers
                )
                for single_waveform in waveform
            ]
            return encoded_batch
        else:
            raise ValueError("waveform must be a torch.Tensor or list of torch.Tensors")

    def _encode(
        self,
        waveform: torch.Tensor,
        original_sampling_rate: int,
        num_quantizers: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Core encoding logic.

        Args:
            waveform: Raw audio waveform [channels, samples]
            original_sampling_rate: Original sampling rate of the audio
            num_quantizers: Number of quantizers to use (uses default if None)

        Returns:
            Encoded audio tokens [batch, quantizers, time_steps]
        """
        if num_quantizers is None:
            num_quantizers = self.num_quantizers

        # Prepare the audio
        prepared_waveform = self._prepare_audio(waveform, original_sampling_rate)

        # The model expects a 1D array, so we squeeze the tensor
        audio_sample = prepared_waveform.squeeze().numpy()

        # Pre-process the inputs
        inputs = self.feature_extractor(
            raw_audio=audio_sample,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Encode the audio
        with torch.no_grad():
            encoder_outputs = self.model.encode(
                inputs["input_values"], num_quantizers=num_quantizers
            )

        return encoder_outputs.audio_codes  # [batch, quantizers, time_steps]

    def decode_to_waveform(
        self, audio_codes: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Decode audio tokens back to raw waveform(s).

        Args:
            audio_codes: Encoded audio tokens [batch, quantizers, time_steps]
                        or list of encoded tensors

        Returns:
            Decoded audio waveform [batch, samples] for single input,
            or list of decoded waveforms for multiple inputs
        """
        if isinstance(audio_codes, torch.Tensor):
            # Single audio codes tensor
            # Ensure audio_codes is on the correct device
            audio_codes = audio_codes.to(self.device)

            with torch.no_grad():
                decoded_outputs = self.model.decode(audio_codes)

            return decoded_outputs.audio_values  # [batch, samples]

        elif isinstance(audio_codes, list):
            # Multiple audio codes tensors
            decoded_batch = []
            for single_codes in audio_codes:
                decoded = self.decode_to_waveform(single_codes)
                decoded_batch.append(decoded)
            return decoded_batch
        else:
            raise ValueError(
                "audio_codes must be a torch.Tensor or list of torch.Tensors"
            )

    def decode_to_file(
        self,
        audio_codes: Union[torch.Tensor, List[torch.Tensor]],
        output_path: Union[str, List[str]],
        format: str = "wav",
    ):
        """
        Decode audio tokens and save to file(s).

        Args:
            audio_codes: Encoded audio tokens [batch, quantizers, time_steps]
                        or list of encoded tensors
            output_path: Path to save the output audio file or list of paths
            format: Audio format to save as
        """
        if isinstance(audio_codes, list):
            if not isinstance(output_path, list):
                # Multiple audio codes and single output path (save as batch with numbered files)
                output_path = [
                    f"{output_path.rsplit('.', 1)[0]}_{i}.{output_path.rsplit('.', 1)[1]}"
                    for i in range(len(audio_codes))
                ]

            if len(audio_codes) != len(output_path):
                raise ValueError(
                    "Number of audio codes must match number of output paths"
                )

            for codes, path in zip(audio_codes, output_path):
                self.decode_to_file(codes, path, format)
        elif isinstance(audio_codes, torch.Tensor):
            if not isinstance(output_path, str):
                raise ValueError("output_path must be a string")

            # Single audio codes and single output path
            decoded_audio = self.decode_to_waveform(audio_codes)

            # Move to CPU for saving
            audio_to_save = decoded_audio.cpu()

            # Handle tensor dimensions for torchaudio.save
            # decoded_audio shape is [batch, channels, samples]
            # For single batch, squeeze the batch dimension
            if audio_to_save.shape[0] == 1:
                audio_to_save = audio_to_save.squeeze(0)  # Remove batch dimension

            # Save the audio file
            torchaudio.save(
                output_path, audio_to_save, self.sampling_rate, format=format
            )
            print(f"Audio saved to {output_path}")
        else:
            raise ValueError(
                "audio_codes must be a torch.Tensor or list of torch.Tensors"
            )

    def encode_decode_cycle(
        self, audio_path: str, num_quantizers: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a complete encode-decode cycle.

        Args:
            audio_path: Path to input audio file
            num_quantizers: Number of quantizers to use (uses default if None)

        Returns:
            Tuple of (original_waveform, decoded_waveform)
        """
        # Load original audio
        original_waveform, original_sr = self.load_audio_from_path(audio_path)
        prepared_original = self._prepare_audio(
            original_waveform, original_sr
        ).squeeze()

        # Encode
        encoded_tokens = self.encode_from_waveform(
            original_waveform, original_sr, num_quantizers
        )

        # Decode
        decoded_waveform = self.decode_to_waveform(encoded_tokens).squeeze()

        return prepared_original, decoded_waveform

    def batch_encode(
        self, audio_paths: list, num_quantizers: Optional[int] = None
    ) -> list:
        """
        Encode multiple audio files in batch.

        Args:
            audio_paths: List of audio file paths
            num_quantizers: Number of quantizers to use (uses default if None)

        Returns:
            List of encoded audio tokens
        """
        encoded_batch = []
        for path in audio_paths:
            encoded = self.encode_from_path(path, num_quantizers)
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
        print(f"Model moved to {device}")

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_string": self.model_string,
            "sampling_rate": self.sampling_rate,
            "num_quantizers": self.num_quantizers,
            "device": self.device,
            "is_cuda_available": torch.cuda.is_available(),
            "model_on_gpu": next(self.model.parameters()).is_cuda
            if hasattr(next(self.model.parameters()), "is_cuda")
            else False,
        }
