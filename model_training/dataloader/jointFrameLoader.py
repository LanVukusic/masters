import os
import torch
from torch.utils.data import Dataset
import gc
import soundfile
import torchcodec
import math
from model_training.tokenizer.tokenizer import AudioTokenizer
from tqdm import tqdm


class JointFrameLoader(Dataset):
    def __init__(
        self,
        num_frames: int,
        tokenizer=AudioTokenizer,  # Can be class or instance
        num_quantizers: int = 6,
        audio_dir: str = ".",
        max_concurrent_frames: int = 3000,  # Max frames to process at once
        cache_size: int = 50,  # Number of *chunks* to cache per worker
    ):
        # Initialize tokenizer (handle both class and instance)
        self.tokenizer = tokenizer

        self.num_quantizers = num_quantizers
        self.frame_size = (
            self.tokenizer.frame_size
        )  # minimal number of samples that tokenizer can process
        self.sampling_rate = self.tokenizer.sampling_rate  # raw audio sampling rate
        self.num_frames = num_frames  # tokenized chunks  return
        self.max_concurrent_frames = max_concurrent_frames  # max song length to not over consume RAM while tokenizing
        self.cache_size = cache_size

        # Calculate total sequence duration
        frame_duration_s = self.frame_size / self.sampling_rate
        total_duration_s_s = num_frames * frame_duration_s

        print(
            f"Dataset configured for {num_frames} chunks of {frame_duration_s:.3f}s each. "
            f"Total sequence: {total_duration_s_s:.3f}s. "
            f"Processing audio in windows of {max_concurrent_frames} frames "
            f"({max_concurrent_frames * frame_duration_s:.3f}s)."
        )

        # Find all audio files and calculate slices
        self.slices = []
        audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

        print("Discovering audio files and calculating slices...")
        for root, _, files in tqdm(os.walk(audio_dir)):
            for file in files:
                if os.path.splitext(file.lower())[1] in audio_extensions:
                    path = os.path.join(root, file)
                    try:
                        # Get audio info without loading the entire file
                        info = soundfile.info(path)
                        # We now calculate slices based on the FULL duration of the file
                        total_samples = int(info.duration * self.sampling_rate)
                        song_frames = int(total_samples // self.frame_size)

                        # Calculate number of possible slices from the full song
                        num_slices = max(0, song_frames - self.num_frames + 1)

                        # Store slices as (start_frame, end_frame, path) tuples
                        for i in range(num_slices):
                            start_frame = i
                            end_frame = i + self.num_frames
                            self.slices.append((start_frame, end_frame, path))
                    except Exception as e:
                        print(f"Warning: Could not process {path}: {e}")
                        continue
        if len(self.slices) == 0:
            raise Exception(f"No slices could be generated from directory {audio_dir}.")

        # Cache and metadata will be initialized per worker
        self.cache = {}
        self.cache_order = []
        self.song_chunk_info = {}  # Stores metadata about chunks for each song

    def __len__(self):
        return len(self.slices)

    def _initialize_worker_cache(self):
        """Initialize worker-specific cache if not already done."""
        if not hasattr(self, "cache") or self.cache is None:
            self.cache = {}
            self.cache_order = []
            self.song_chunk_info = {}

    def _process_and_cache_song_chunks(self, path):
        """
        Loads an audio file, processes it in chunks, and caches the encoded chunks.
        This method is idempotent; it will only process a song once per worker.
        """
        if path in self.song_chunk_info:
            return  # Already processed and cached for this worker

        try:
            # Load the entire audio waveform. The waveform itself is smaller than the
            # encoded output, so loading it all at once is acceptable.
            song_waveform = (
                torchcodec.decoders.AudioDecoder(
                    source=path,
                    sample_rate=self.sampling_rate,
                    num_channels=1,
                )
                .get_all_samples()
                .data
            )

            total_samples = song_waveform.shape[0]
            total_frames = int(total_samples // self.frame_size)

            # Calculate number of chunks needed
            num_chunks = math.ceil(total_frames / self.max_concurrent_frames)
            self.song_chunk_info[path] = {"num_chunks": num_chunks}

            print(f"Processing {num_chunks} chunks for {os.path.basename(path)}...")

            # split the song into chunks that we can actually process on our hardware
            for i in range(num_chunks):
                # Define frame boundaries for this chunk
                start_frame = i * self.max_concurrent_frames
                end_frame = min((i + 1) * self.max_concurrent_frames, total_frames)

                # Convert frame boundaries to sample boundaries
                start_sample = start_frame * self.frame_size
                end_sample = end_frame * self.frame_size

                # Slice the waveform
                waveform_chunk = song_waveform[start_sample:end_sample]

                # Encode the chunk
                # Shape: [num_quantizers, chunk_num_frames]
                encoded_chunk = self.tokenizer.encode_from_waveform(
                    waveform_chunk,
                    self.sampling_rate,
                )

                # Implement LRU cache for chunks
                if len(self.cache) >= self.cache_size:
                    oldest_key = self.cache_order.pop(0)
                    del self.cache[oldest_key]

                # Cache the chunk with a composite key
                cache_key = (path, i)
                self.cache[cache_key] = encoded_chunk
                self.cache_order.append(cache_key)

            # Periodic garbage collection to free memory
            gc.collect()

        except Exception as e:
            print(f"Error during chunked processing of {path}: {e}")
            # Mark as processed to avoid retrying, but with zero chunks
            self.song_chunk_info[path] = {"num_chunks": 0}

    def __getitem__(self, idx):
        """
        Get a slice of encoded audio frames, handling cross-chunk concatenation.

        Returns:
            torch.Tensor: Shape [num_quantizers, num_frames]
        """
        self._initialize_worker_cache()

        start_frame, end_frame, path = self.slices[idx]

        # Ensure the song's chunks are processed and cached
        self._process_and_cache_song_chunks(path)

        # Determine which chunks are needed
        start_chunk_idx = start_frame // self.max_concurrent_frames
        end_chunk_idx = (
            end_frame - 1
        ) // self.max_concurrent_frames  # -1 for inclusive calculation

        collected_chunks = []

        for i in range(start_chunk_idx, end_chunk_idx + 1):
            cache_key = (path, i)
            if cache_key not in self.cache:
                # This should not happen if _process_and_cache_song_chunks worked
                raise Exception("Despite forcing cache, {cache_key} is missing. Panic!")

            chunk = self.cache[cache_key]

            # Determine the slice for the current chunk
            chunk_start_frame = i * self.max_concurrent_frames
            chunk_end_frame = (i + 1) * self.max_concurrent_frames

            # Calculate local start/end indices within this chunk
            local_start = max(0, start_frame - chunk_start_frame)
            local_end = min(chunk.shape[1], end_frame - chunk_start_frame)

            collected_chunks.append(chunk[:, local_start:local_end])

        # Concatenate the collected chunks
        if collected_chunks:
            final_tensor = torch.cat(collected_chunks, dim=1)
            # Ensure the output has the exact number of frames requested
            if final_tensor.shape[1] == self.num_frames:
                return final_tensor
            else:
                # Pad with zeros if something went wrong (e.g., song shorter than expected)
                padding = torch.zeros(
                    (self.num_quantizers, self.num_frames - final_tensor.shape[1]),
                    dtype=torch.long,
                )
                return torch.cat([final_tensor, padding], dim=1)
        else:
            # Return zeros if no chunks were found
            return torch.zeros((self.num_quantizers, self.num_frames), dtype=torch.long)
