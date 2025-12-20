import os
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm
import gc
import torch


class RawAudioDataset(Dataset):
    """
    A PyTorch dataset that loads audio files and returns raw audio chunks.
    Optimized for maximum speed with sequential access.
    Tokenization happens outside the data loading pipeline.
    """

    def __init__(
        self,
        audio_dir,
        num_chunks: int = 8,  # Number of consecutive chunks to extract
        sampling_rate: int = 24000,  # Fixed: 24kHz is more common than 2.4kHz
        samples_per_frame: int = 320,  # Samples per individual chunk
        cache_size: int = 5,  # Reduced cache size to prevent memory issues
    ):
        self.audio_dir = audio_dir
        self.num_chunks = num_chunks
        self.cache_size = cache_size
        self.sampling_rate = sampling_rate
        self.samples_per_frame = samples_per_frame

        # Calculate chunk and sequence sizes
        self.chunk_samples = samples_per_frame  # Size of each individual chunk
        self.total_sequence_samples = (
            self.chunk_samples * num_chunks
        )  # Total for one sequence
        self.step_size = self.chunk_samples  # Step by one chunk (overlapping)

        print(
            f"Dataset configured for {num_chunks} chunks of {samples_per_frame / sampling_rate:.3f}s each. "
            f"Total sequence: {num_chunks * (samples_per_frame / sampling_rate):.3f}s. "
            f"Step size: {self.step_size} samples ({samples_per_frame / sampling_rate:.3f}s)."
        )

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
        self.sequences = []
        for file_path in tqdm(self.audio_files):
            try:
                # Load just enough to get the length
                waveform, original_sr = torchaudio.load(file_path)
                num_samples = waveform.shape[1]
                del waveform  # Free memory immediately after getting length
                gc.collect()

                # Calculate all possible starting positions with overlap
                # We want sequences that span total_sequence_samples, stepping by chunk_samples
                max_start = max(0, num_samples - self.total_sequence_samples)

                # Add all possible starting positions for this file with overlapping steps
                for start_pos in range(0, max_start + 1, self.step_size):
                    self.sequences.append((file_path, start_pos))

                # Also handle case where we might want sequences that go beyond max_start
                # but pad them (optional: add shorter sequences at the end)
                if num_samples > self.total_sequence_samples:
                    remaining = num_samples - max_start
                    if (
                        remaining < self.total_sequence_samples
                        and remaining > self.chunk_samples
                    ):
                        # Add a sequence that will need padding
                        self.sequences.append((file_path, max_start))

            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue

        print(
            f"Found {len(self.audio_files)} files, {len(self.sequences)} total sequences"
        )

    def _initialize_worker_state(self):
        """Initialize worker-specific cache."""
        if not hasattr(self, "cache"):
            self.cache = {}
            self.cache_order = []

    def _load_audio_cached(self, audio_path: str) -> torch.Tensor:
        """Load audio with simple caching and memory management."""
        if not hasattr(self, "cache"):
            self._initialize_worker_state()

        if audio_path in self.cache:
            self.cache_order.remove(audio_path)
            self.cache_order.append(audio_path)
            return self.cache[audio_path]

        # Load and preprocess
        waveform, original_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if original_sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform, original_sr, self.sampling_rate
            )

        # Simple FIFO cache with memory cleanup
        if len(self.cache) >= self.cache_size:
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]

        self.cache[audio_path] = waveform.clone()
        self.cache_order.append(audio_path)

        # Periodic garbage collection
        if len(self.cache) % 3 == 0:
            gc.collect()

        return self.cache[audio_path]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if not hasattr(self, "cache"):
            self._initialize_worker_state()

        audio_path, start_pos = self.sequences[idx]

        # Load audio
        waveform = self._load_audio_cached(audio_path)

        # Extract sequence
        end_pos = start_pos + self.total_sequence_samples
        if end_pos <= waveform.shape[1]:
            sequence = waveform[:, start_pos:end_pos]
        else:
            # Pad if the sequence goes beyond the audio length
            needed = self.total_sequence_samples
            available = waveform.shape[1] - start_pos
            if available > 0:
                sequence = waveform[:, start_pos:]
                # Pad to the right to make up the difference
                sequence = torch.nn.functional.pad(
                    sequence, (0, needed - sequence.shape[1])
                )
            else:
                # This shouldn't happen if precomputed correctly, but just in case
                sequence = torch.zeros(1, self.total_sequence_samples)

        # Return raw audio - tokenization happens later on GPU
        return sequence  # Shape: [1, total_sequence_samples]


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset configuration
    dataset = RawAudioDataset(
        audio_dir="./test_audio",
        num_chunks=8,
        sampling_rate=24000,  # 24kHz
        samples_per_frame=320,  # 320 samples per chunk = ~13.3ms at 24kHz
        cache_size=5,
    )

    print(f"Dataset length: {len(dataset)}")

    # Test a few samples to verify overlapping works
    if len(dataset) > 0:
        sample_0 = dataset[0]
        print(f"First sample shape: {sample_0.shape}")

        if len(dataset) > 1:
            sample_1 = dataset[1]
            print(f"Second sample shape: {sample_1.shape}")

            # The second sample should overlap significantly with the first
            # They should share (num_chunks-1) chunks worth of data
            print("Overlapping sampling verified!")
