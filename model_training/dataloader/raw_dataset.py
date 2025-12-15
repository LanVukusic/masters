import os
import torch
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

# Note: Importing tokenizer classes here is okay for initialization info,
# but we won't create the actual models in the dataset workers
from model_training.tokenizer.mimi_audio_tokenizer import (
    MimiAudioTokenizer,
    SAMPLES_PER_FRAME as mimi_samples_per_frame,
)
from model_training.tokenizer.dac_audio_tokenizer import (
    DACAudioTokenizer,
    SAMPLES_PER_FRAME as dac_samples_per_frame,
)


class RawAudioDataset(Dataset): # Renamed to reflect its function
    """
    A PyTorch dataset that loads audio files and returns raw audio chunks.
    Optimized for maximum speed with sequential access.
    Tokenization happens outside the data loading pipeline.
    """

    def __init__(
        self,
        audio_dir,
        tokenizer_type: str = "DAC",  # "DAC" or "MIMI" - only for getting sampling rate
        num_chunks: int = 8,  # Number of consecutive chunks to extract
        chunk_duration: float = 2.0,  # in seconds
        cache_size: int = 10, # Cache raw audio files
    ):
        self.audio_dir = audio_dir
        self.num_chunks = num_chunks
        self.cache_size = cache_size

        # Determine sampling rate based on tokenizer type
        if tokenizer_type == "DAC":
            temp_tokenizer = DACAudioTokenizer(num_quantizers=8, device="cpu") # Temp instance to get sampling rate
            self.sampling_rate = temp_tokenizer.sampling_rate
            self.samples_per_frame = dac_samples_per_frame
            del temp_tokenizer
        else: # MIMI
            temp_tokenizer = MimiAudioTokenizer(num_quantizers=8, device="cpu") # Temp instance to get sampling rate
            self.sampling_rate = temp_tokenizer.sampling_rate
            self.samples_per_frame = mimi_samples_per_frame
            del temp_tokenizer

        # Calculate chunk and sequence sizes
        chunk_samples = int(chunk_duration * self.sampling_rate)
        self.chunk_size_samples = (chunk_samples // self.samples_per_frame) * self.samples_per_frame
        total_samples = int(num_chunks * chunk_duration * self.sampling_rate)
        self.total_sequence_samples = (total_samples // self.samples_per_frame) * self.samples_per_frame

        # Find all audio files
        self.audio_files = []
        audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if os.path.splitext(file.lower())[1] in audio_extensions:
                    self.audio_files.append(os.path.join(root, file))

        if not self.audio_files:
            raise ValueError(f"No audio files found in directory: {audio_dir}")

        # Precompute all possible sequence positions across all files
        self.sequences = []
        for file_path in tqdm(self.audio_files):
            try:
                # Load just enough to get the length - this is the most reliable method now
                waveform, original_sr = torchaudio.load(file_path)
                num_samples = waveform.shape[1]

                max_start = max(0, num_samples - self.total_sequence_samples)

                # Add all possible starting positions for this file
                step_size = self.chunk_size_samples  # Step by one chunk
                for start_pos in range(0, max_start + 1, step_size):
                    self.sequences.append((file_path, start_pos))
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue

        print(f"Found {len(self.audio_files)} files, {len(self.sequences)} total sequences")

    def _initialize_worker_state(self):
        """Initialize worker-specific cache."""
        if not hasattr(self, 'cache'):
            self.cache = {}

    def _load_audio_cached(self, audio_path: str) -> torch.Tensor:
        """Load audio with simple caching."""
        if not hasattr(self, 'cache'):
            self._initialize_worker_state()

        if audio_path in self.cache:
            return self.cache[audio_path]

        # Load and preprocess
        waveform, original_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, original_sr, self.sampling_rate)

        # Simple FIFO cache
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[audio_path] = waveform

        return waveform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if not hasattr(self, 'cache'):
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

        # Return raw audio - tokenization happens later on GPU
        return sequence # Shape: [1, total_sequence_samples]
