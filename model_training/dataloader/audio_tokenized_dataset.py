import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from collections import OrderedDict
import threading
from typing import Optional

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
    A PyTorch dataset that loads audio files and returns tokenized chunks using AudioTokenizer.
    Implements caching to avoid repeated audio loading.
    """

    def __init__(
        self,
        audio_dir,
        tokenizer: "DAC", # DAC or MIMI
        num_chunks: int = 8,
        rvq_depth: int = 8,
        chunk_duration: float = 2.0,  # in seconds
        max_samples_per_file: int = 10,  # Maximum number of random samples per file to avoid memory issues
        cache_size: int = 10,  # Maximum number of songs to keep in cache
        preload_cache: bool = False,  # Whether to preload songs into cache on initialization
        device: str = "cpu"
    ):
        """
        Args:
            audio_dir: Directory containing .mp3 files
            tokenizer: AudioTokenizer instance for encoding
            num_chunks: Number of chunks to extract from each sample
            rvq_depth: Number of quantizers (RVQ depth) to use
            chunk_duration: Duration of each chunk in seconds
            max_samples_per_file: Maximum random samples per file to avoid infinite loops
            cache_size: Maximum number of songs to cache in memory
            preload_cache: Whether to preload songs into cache during initialization
        """
        self.audio_dir = audio_dir
        self.num_chunks = num_chunks
        self.rvq_depth = rvq_depth
        self.chunk_duration = chunk_duration
        self.max_samples_per_file = max_samples_per_file
        self.cache_size = cache_size

        # Create tokenizer based on selection
        if  tokenizer == "DAC":
            self.tokenizer = DACAudioTokenizer(
                num_quantizers = rvq_depth,
                device=device,
            )
            self.samples_per_frame = dac_samples_per_frame
         
        else:
            self.tokenizer = MimiAudioTokenizer(
                num_quantizers = rvq_depth,
                device=device,
            )
            self.samples_per_frame = mimi_samples_per_frame


        self.sampling_rate = self.tokenizer.sampling_rate
        print(f"Using {tokenizer} tokenizer with {self.rvq_depth} quantizers and sampling rate: {self.sampling_rate}")
        with torch.no_grad():
            test_audio = torch.zeros(1, self.tokenizer.sampling_rate)  # 1 second of zeros at 24kHz
            encoded_test = self.tokenizer.encode_from_waveform(test_audio, self.tokenizer.sampling_rate)
            actual_quantizers = encoded_test.shape[1]  # Get quantizer dimension
            print(f"Actual quantizers : {actual_quantizers}; shape: {encoded_test.shape}")
        
      
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

        print(f"Number of audio files found: {len(self.audio_files)}")

        # Initialize cache - using OrderedDict for LRU behavior
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()
        
        # Initialize per-file metadata and state
        self.file_metadata = {}
        self.sample_positions = []
        
        # Create sample positions without needing metadata
        for audio_path in self.audio_files:
            # Each file gets max_samples_per_file sample positions
            for i in range(max_samples_per_file):
                self.sample_positions.append((audio_path, i))
        
        # Shuffle sample positions for randomness
        random.shuffle(self.sample_positions)
        
        print(f"Total samples in dataset: {len(self.sample_positions)}")
        
        # Preload cache if requested
        if preload_cache:
            self._preload_cache()

    def _preload_cache(self):
        """Preload songs into cache."""
        # Preload up to cache_size songs
        files_to_load = min(self.cache_size, len(self.audio_files))
        for i in range(files_to_load):
            audio_path = self.audio_files[i]
            self._load_and_cache_audio(audio_path)
            print(f"Preloaded {i+1}/{files_to_load}: {os.path.basename(audio_path)}")

    def _load_and_cache_audio(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file, resample to target rate, and cache it.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Resampled waveform tensor
        """
        with self.cache_lock:
            # Check if already in cache
            if audio_path in self.cache:
                # Move to end to mark as recently used
                waveform = self.cache.pop(audio_path)
                self.cache[audio_path] = waveform
                return waveform
            
            # Load audio file
            try:
                waveform, original_sr = torchaudio.load(audio_path)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                # Return a silent waveform of minimum required length
                waveform = torch.zeros(1, self.total_sequence_samples)
                self.cache[audio_path] = waveform
                return waveform
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if original_sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sr, new_freq=self.sampling_rate
                )
                waveform = resampler(waveform)
            
            # Cache the waveform
            if len(self.cache) >= self.cache_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
            
            self.cache[audio_path] = waveform
            return waveform

    def _get_cached_waveform(self, audio_path: str) -> torch.Tensor:
        """
        Get waveform from cache, loading if not present.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Cached waveform tensor
        """
        return self._load_and_cache_audio(audio_path)

    def _extract_random_sequence(self, waveform: torch.Tensor, sample_idx: int) -> torch.Tensor:
        """
        Extract a random sequence from the waveform.
        
        Args:
            waveform: Waveform tensor of shape [1, samples]
            sample_idx: Index of the sample to extract (for deterministic randomness)
            
        Returns:
            Extracted sequence of shape [1, total_sequence_samples]
        """
        # Use sample_idx as a seed for deterministic randomness per sample position
        rng = random.Random(hash(str(sample_idx)) % (2**32))
        
        # If the waveform is shorter than needed, pad it
        if waveform.shape[1] < self.total_sequence_samples:
            # If the file is too short, we need to handle it
            if waveform.shape[1] < self.total_sequence_samples:
                # Pad with zeros
                padding_needed = self.total_sequence_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
                return waveform
        
        # Randomly select a starting point
        max_start = waveform.shape[1] - self.total_sequence_samples
        if max_start > 0:
            start_idx = rng.randint(0, max_start)
            return waveform[:, start_idx:start_idx + self.total_sequence_samples]
        else:
            # If the file is exactly the right length or shorter
            return waveform[:, :self.total_sequence_samples]

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.sample_positions)

    def __getitem__(self, idx):
        """
        Get a tokenized chunk from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tokenized audio chunk as tensor of shape [rvq_depth, num_time_steps]
        """
        if idx >= len(self.sample_positions):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.sample_positions)}")
        
        # Get the file and sample index
        audio_path, sample_idx = self.sample_positions[idx]
        
        # Get waveform from cache (loads if not cached)
        waveform = self._get_cached_waveform(audio_path)
        
        # Extract random sequence
        audio_sequence = self._extract_random_sequence(waveform, sample_idx)
        
        # Tokenize the entire sequence at once
        with torch.no_grad():
            # Encode using the specified RVQ depth
            # Check if the tokenizer supports num_quantizers parameter (Mimi) or not (DAC)
            import inspect
            sig = inspect.signature(self.tokenizer.encode_from_waveform)
            if 'num_quantizers' in sig.parameters:
                # Mimi tokenizer - supports num_quantizers
                encoded_tokens = self.tokenizer.encode_from_waveform(
                    audio_sequence, self.sampling_rate, num_quantizers=self.rvq_depth
                )
            else:
                # DAC tokenizer - doesn't support num_quantizers, uses all quantizers
                encoded_tokens = self.tokenizer.encode_from_waveform(
                    audio_sequence, self.sampling_rate
                )
            
            # The encoded_tokens shape should be [batch, quantizers, time_steps]
            # Since we're processing one sequence, it should be [1, rvq_depth, time_steps]
            if len(encoded_tokens.shape) == 3:
                encoded_tokens = encoded_tokens.squeeze(0)  # Remove batch dimension
        
        return encoded_tokens

    def clear_cache(self):
        """Clear the audio cache to free memory."""
        with self.cache_lock:
            self.cache.clear()

    def get_cache_info(self):
        """Get information about the cache."""
        with self.cache_lock:
            return {
                'size': len(self.cache),
                'cache_size': self.cache_size,
                'cached_files': list(self.cache.keys()),
                'memory_usage_mb': sum(
                    w.element_size() * w.nelement() / (1024 * 1024)
                    for w in self.cache.values()
                ) if self.cache else 0,
            }
