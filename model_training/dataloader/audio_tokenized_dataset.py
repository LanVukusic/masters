import os
import torch
import torchaudio
from torch.utils.data import Dataset

from model_training.tokenizer.mimi_audio_tokenizer import (
    MimiAudioTokenizer,
    SAMPLES_PER_FRAME as mimi_samples_per_frame,
)
from model_training.tokenizer.dac_audio_tokenizer import (
    DACAudioTokenizer,
    SAMPLES_PER_FRAME as dac_samples_per_frame,
)


class AudioTokenizedDataset(Dataset):
    """
    A PyTorch dataset that loads audio files and returns sequential tokenized chunks.
    Optimized for maximum speed with sequential access.
    """

    def __init__(
        self,
        audio_dir,
        tokenizer_type: str = "DAC",  # "DAC" or "MIMI"
        num_chunks: int = 8,  # Number of consecutive chunks to extract
        rvq_depth: int = 8,
        chunk_duration: float = 2.0,  # in seconds
        cache_size: int = 10,
        device: str = "cpu",
    ):
        self.audio_dir = audio_dir
        self.num_chunks = num_chunks
        self.rvq_depth = rvq_depth
        self.cache_size = cache_size
        self.device = device

        # Store tokenizer class info
        self.tokenizer_type = tokenizer_type
        if tokenizer_type == "DAC":
            self.tokenizer_class = DACAudioTokenizer
            self.samples_per_frame = dac_samples_per_frame
        else:
            self.tokenizer_class = MimiAudioTokenizer
            self.samples_per_frame = mimi_samples_per_frame

        # Get sampling rate
        temp_tokenizer = self.tokenizer_class(num_quantizers=rvq_depth, device=device)
        self.sampling_rate = temp_tokenizer.sampling_rate
        del temp_tokenizer

        # Calculate chunk and sequence sizes
        chunk_samples = int(chunk_duration * self.sampling_rate)
        self.chunk_size_samples = (
            chunk_samples // self.samples_per_frame
        ) * self.samples_per_frame
        total_samples = int(num_chunks * chunk_duration * self.sampling_rate)
        self.total_sequence_samples = (
            total_samples // self.samples_per_frame
        ) * self.samples_per_frame

        # Find all audio files
        self.audio_files = []
        audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in audio_extensions:
                    self.audio_files.append(os.path.join(root, file))

        if not self.audio_files:
            raise ValueError(f"No audio files found in directory: {audio_dir}")

        # Precompute all possible sequence positions across all files
        # To avoid torchaudio.info (deprecated), we'll store basic metadata
        # and calculate sequence positions during first access if needed
        self.file_metadata = []
        for file_path in self.audio_files:
            try:
                # Load just enough to get the length - this is the most reliable method now
                waveform, original_sr = torchaudio.load(file_path)
                num_samples = waveform.shape[1]

                max_start = max(0, num_samples - self.total_sequence_samples)

                # Store metadata for this file
                self.file_metadata.append(
                    {
                        "path": file_path,
                        "num_samples": num_samples,
                        "max_start": max_start,
                    }
                )
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue

        # Now create all sequence positions
        self.sequences = []
        for meta in self.file_metadata:
            step_size = self.chunk_size_samples  # Step by one chunk
            for start_pos in range(0, meta["max_start"] + 1, step_size):
                self.sequences.append((meta["path"], start_pos))

        print(
            f"Found {len(self.audio_files)} files, {len(self.sequences)} total sequences"
        )

    def _initialize_worker_state(self):
        """Initialize worker-specific state."""
        if not hasattr(self, "tokenizer"):
            self.tokenizer = self.tokenizer_class(
                num_quantizers=self.rvq_depth, device=self.device
            )
            self.cache = {}

    def _load_audio_cached(self, audio_path: str) -> torch.Tensor:
        """Load audio with simple caching."""
        if not hasattr(self, "cache"):
            self._initialize_worker_state()

        if audio_path in self.cache:
            return self.cache[audio_path]

        # Load and preprocess
        waveform, original_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform, original_sr, self.sampling_rate
            )

        # Simple FIFO cache
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[audio_path] = waveform

        return waveform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if not hasattr(self, "tokenizer"):
            self._initialize_worker_state()

        audio_path, start_pos = self.sequences[idx]

        # Load audio
        waveform = self._load_audio_cached(audio_path)

        # Extract sequence
        end_pos = start_pos + self.total_sequence_samples
        if end_pos <= waveform.shape[1]:
            sequence = waveform[:, start_pos:end_pos]
        else:
            # Pad if needed
            needed = self.total_sequence_samples
            available = waveform.shape[1] - start_pos
            if available > 0:
                sequence = waveform[:, start_pos:]
                sequence = torch.nn.functional.pad(sequence, (0, needed - available))
            else:
                sequence = torch.zeros(1, self.total_sequence_samples)

        # Tokenize
        with torch.no_grad():
            encoded = self.tokenizer.encode_from_waveform(sequence, self.sampling_rate)
            if encoded.dim() == 3:
                encoded = encoded.squeeze(0)

        return encoded
