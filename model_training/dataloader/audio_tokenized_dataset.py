import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from model_training.tokenizer.audio_tokenizer import (
    AudioTokenizer,
    SAMPLES_PER_FRAME,
    SAMPLE_RATE,
)


class AudioTokenizedDataset(Dataset):
    """
    A PyTorch dataset that loads audio files and returns tokenized chunks using AudioTokenizer.
    Each audio file is sampled multiple times to maximize dataset usage.
    """

    def __init__(
        self,
        audio_dir,
        tokenizer: AudioTokenizer,
        num_chunks: int = 8,
        rvq_depth: int = 8,
        chunk_duration: float = 2.0,  # in seconds
        max_samples_per_file: int = 10,  # Maximum number of random samples per file to avoid memory issues
    ):
        """
        Args:
            audio_dir: Directory containing .mp3 files
            tokenizer: AudioTokenizer instance for encoding
            num_chunks: Number of chunks to extract from each sample
            rvq_depth: Number of quantizers (RVQ depth) to use
            chunk_duration: Duration of each chunk in seconds
            max_samples_per_file: Maximum random samples per file to avoid infinite loops
        """
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.num_chunks = num_chunks
        self.rvq_depth = rvq_depth
        self.chunk_duration = chunk_duration
        self.max_samples_per_file = max_samples_per_file

        # Use the tokenizer's native sampling rate
        self.sampling_rate = SAMPLE_RATE  # 24000 Hz
        self.samples_per_frame = SAMPLES_PER_FRAME  # 1920 samples per frame

        # Calculate chunk size in samples (must be multiple of frame size for proper tokenization)
        chunk_samples = int(chunk_duration * self.sampling_rate)
        # Round to nearest multiple of frame size for proper tokenization
        self.chunk_size_samples = (
            chunk_samples // self.samples_per_frame
        ) * self.samples_per_frame

        # Calculate total sequence size for multiple chunks
        total_samples = int(self.num_chunks * chunk_duration * self.sampling_rate)
        self.total_sequence_samples = (
            total_samples // self.samples_per_frame
        ) * self.samples_per_frame

        # Find all MP3 files
        self.audio_files = []
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.lower().endswith(".mp3"):
                    self.audio_files.append(os.path.join(root, file))

        if not self.audio_files:
            raise ValueError(f"No .mp3 files found in directory: {audio_dir}")

        # Calculate pessimistic estimate of total samples
        # For each file, calculate how many complete tokenizable sequences we can extract
        # Each sequence is num_chunks * chunk_duration seconds long
        self.total_estimated_samples = 0
        for audio_path in self.audio_files:
            try:
                waveform, original_sr = torchaudio.load(audio_path)

                # Resample to tokenizer's sampling rate for accurate calculation
                if original_sr != self.sampling_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=original_sr, new_freq=self.sampling_rate
                    )
                    waveform = resampler(waveform)

                # Calculate how many complete frames we have
                total_frames = waveform.shape[1] // self.samples_per_frame

                # Calculate how many frames we need for one complete sequence
                frames_per_sequence = (
                    self.total_sequence_samples // self.samples_per_frame
                )

                # Calculate how many complete sequences we can fit
                if frames_per_sequence > 0:
                    sequences_per_file = max(0, total_frames - frames_per_sequence + 1)
                    # Multiply by max_samples_per_file to account for random sampling
                    self.total_estimated_samples += min(
                        sequences_per_file, self.max_samples_per_file
                    )
                else:
                    self.total_estimated_samples += self.max_samples_per_file

            except:
                # If we can't load the file, estimate conservatively
                self.total_estimated_samples += self.max_samples_per_file

    def __len__(self):
        """Return pessimistic estimate of dataset size."""
        return self.total_estimated_samples

    def _get_random_chunk_from_file(self, audio_path: str) -> torch.Tensor:
        """
        Extract a random chunk of specified duration from the audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio chunk as tensor of shape [1, chunk_size_samples]
        """
        # Load audio file
        waveform, original_sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if original_sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=self.sampling_rate
            )
            waveform = resampler(waveform)

        # Check if the audio is long enough for the requested chunk size
        if waveform.shape[1] < self.chunk_size_samples:
            # Pad with zeros if too short
            padding_needed = self.chunk_size_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
            return waveform[:, : self.chunk_size_samples]

        # Randomly select a starting point for the chunk
        max_start = waveform.shape[1] - self.chunk_size_samples
        start_idx = random.randint(0, max_start)
        chunk = waveform[:, start_idx : start_idx + self.chunk_size_samples]

        return chunk

    def _extract_multiple_chunks_from_file(self, audio_path: str) -> torch.Tensor:
        """
        Extract multiple sequential chunks from an audio file to fill the desired number.

        Args:
            audio_path: Path to audio file

        Returns:
            Tensor containing multiple chunks concatenated, with length adjusted to frame size
        """
        # Load audio file
        waveform, original_sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if original_sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=self.sampling_rate
            )
            waveform = resampler(waveform)

        # Calculate total samples needed (rounded to frame size multiple)
        total_samples = self.total_sequence_samples

        # If the file is shorter than needed, pad it to at least the required length
        if waveform.shape[1] < total_samples:
            # If the file is too short, pad it
            if waveform.shape[1] < total_samples:
                padding_needed = total_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
            else:
                # If still not enough, repeat the audio
                while waveform.shape[1] < total_samples:
                    waveform = torch.cat([waveform, waveform], dim=1)

        # Randomly select a starting point to extract the sequence of chunks
        # Ensure we have enough samples after the start point
        max_start = waveform.shape[1] - total_samples
        start_idx = random.randint(0, max_start)
        audio_sequence = waveform[:, start_idx : start_idx + total_samples]

        return audio_sequence

    def __getitem__(self, idx):
        """
        Get a tokenized chunk from the dataset.

        Args:
            idx: Index (used for random selection of audio file and chunk position)

        Returns:
            Tokenized audio chunk as tensor of shape [rvq_depth, num_time_steps]
        """
        # Select a random audio file
        audio_path = random.choice(self.audio_files)

        # Extract multiple chunks worth of audio from the file
        audio_sequence = self._extract_multiple_chunks_from_file(audio_path)

        # Tokenize the entire sequence at once
        with torch.no_grad():
            # Encode using the specified RVQ depth
            encoded_tokens = self.tokenizer.encode_from_waveform(
                audio_sequence, self.sampling_rate, num_quantizers=self.rvq_depth
            )

            # The encoded_tokens shape should be [batch, quantizers, time_steps]
            # Since we're processing one sequence, it should be [1, rvq_depth, time_steps]
            if len(encoded_tokens.shape) == 3:
                encoded_tokens = encoded_tokens.squeeze(0)  # Remove batch dimension

        return encoded_tokens
