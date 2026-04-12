import os
import torch
from torchcodec.decoders import AudioDecoder
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Iterator
import random


class RawAudioDataset(IterableDataset):
    """
    Optimized IterableDataset using torchcodec for loading and resampling.
    - Loads audio using torchcodec AudioDecoder
    - Resamples to target_sampling_rate via sample_rate parameter
    - Output shape: [1, total_sequence_samples] at target_sampling_rate
    """

    def __init__(
        self,
        audio_dir: str,
        num_chunks: int = 8,
        target_sampling_rate: int = 24000,
        samples_per_frame: int = 320,
        cache_size: int = 3,
        shuffle: bool = False,
        device: str = "cpu",
    ):
        self.audio_dir = audio_dir
        self.num_chunks = num_chunks
        self.target_sampling_rate = target_sampling_rate
        self.samples_per_frame = samples_per_frame
        self.shuffle = shuffle
        self.device = device

        self.chunk_samples = samples_per_frame
        self.total_sequence_samples = self.chunk_samples * num_chunks
        self.step_size = self.chunk_samples

        self.audio_files = self._scan_files(audio_dir)
        if not self.audio_files:
            raise ValueError(f"No audio files found in: {audio_dir}")

        self._decoder_cache = {}
        self._sr_cache = {}

        print(
            f"Dataset configured for {num_chunks} chunks of {samples_per_frame / target_sampling_rate:.3f}s each. "
            f"Found {len(self.audio_files)} files. Target sampling rate: {target_sampling_rate}Hz. "
            f"Using torchcodec for loading."
        )

    def _scan_files(self, root: str) -> List[str]:
        extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
        files = []
        for r, _, fs in os.walk(root):
            for f in fs:
                if os.path.splitext(f.lower())[1] in extensions:
                    files.append(os.path.join(r, f))
        return sorted(files)

    def _get_decoder(self, path: str):
        """Get cached decoder or create new one."""
        if path not in self._decoder_cache:
            self._decoder_cache[path] = AudioDecoder(
                path, sample_rate=self.target_sampling_rate
            )
        return self._decoder_cache[path]

    def _get_native_sr(self, path: str) -> int:
        """Get native sample rate of the file."""
        if path not in self._sr_cache:
            decoder = AudioDecoder(path)
            self._sr_cache[path] = decoder.metadata.sample_rate
        return self._sr_cache[path]

    def _get_duration_samples(self, path: str) -> int:
        """Get total samples at native sample rate."""
        decoder = self._get_decoder(path)
        return int(decoder.metadata.duration_seconds * decoder.metadata.sample_rate)

    def _load_chunk(self, path: str, offset: int, native_sr: int) -> torch.Tensor:
        """
        Load a chunk at native sample rate and resample to target rate.

        Args:
            path: Path to audio file
            offset: Offset in samples at native sample rate
            native_sr: Native sample rate of the file

        Returns:
            Tensor of shape [1, total_sequence_samples] at target_sampling_rate
        """
        start_seconds = offset / native_sr
        duration_seconds = self.total_sequence_samples / self.target_sampling_rate

        decoder = self._get_decoder(path)
        samples = decoder.get_samples_played_in_range(
            start_seconds=start_seconds, stop_seconds=start_seconds + duration_seconds
        )

        waveform = samples.data

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            waveform = waveform.unsqueeze(0)

        if waveform.shape[1] < self.total_sequence_samples:
            pad = self.total_sequence_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.shape[1] > self.total_sequence_samples:
            waveform = waveform[:, : self.total_sequence_samples]

        return waveform

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            per_worker = max(1, len(self.audio_files) // num_workers)
            start = worker_id * per_worker
            end = (
                start + per_worker
                if worker_id < num_workers - 1
                else len(self.audio_files)
            )
            worker_files = self.audio_files[start:end]
        else:
            worker_files = self.audio_files

        if self.shuffle:
            random.shuffle(worker_files)

        for file_path in worker_files:
            try:
                native_sr = self._get_native_sr(file_path)
                num_samples = self._get_duration_samples(file_path)

                max_start = max(0, num_samples - self.total_sequence_samples)
                positions = list(range(0, max_start + 1, self.step_size))

                if num_samples > self.total_sequence_samples:
                    remaining = num_samples - max_start
                    if 0 < remaining < self.total_sequence_samples:
                        positions.append(max_start)

                for start_pos in positions:
                    yield self._load_chunk(file_path, start_pos, native_sr)

            except Exception as e:
                print(f"Warning: Skipping {file_path}: {e}")
                continue


if __name__ == "__main__":
    import time

    dataset = RawAudioDataset(
        audio_dir="dataset_gen/rotormotor/mp3s_small", num_chunks=8
    )
    start = time.time()
    for i, sample in enumerate(dataset):
        if i >= 100:
            break
    print(f"100 samples: {time.time() - start:.2f}s")
