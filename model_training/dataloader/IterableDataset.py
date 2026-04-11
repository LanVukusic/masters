import os
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Iterator
import random
import soundfile as sf
import numpy as np

import warnings
warnings.filterwarnings("ignore", message=".*Xing stream size.*")


class RawAudioDataset(IterableDataset):
    """
    Optimized IterableDataset using soundfile.
    - No torchaudio dependency.
    - No torchcodec dependency.
    - Drop-in replacement for original RawAudioDataset.
    - Output shape: [1, total_sequence_samples] (identical to original).
    - Note: Reads at native sample rate, assumes tokenizer will resample if needed.
    """

    def __init__(
        self,
        audio_dir: str,
        num_chunks: int = 8,
        sampling_rate: int = 24000,
        samples_per_frame: int = 320,
        cache_size: int = 3, # ignored
        shuffle: bool = False,
        device: str = "cpu",
    ):
        self.audio_dir = audio_dir
        self.num_chunks = num_chunks
        self.sampling_rate = sampling_rate
        self.samples_per_frame = samples_per_frame
        self.shuffle = shuffle
        self.device = device

        self.chunk_samples = samples_per_frame
        self.total_sequence_samples = self.chunk_samples * num_chunks
        self.step_size = self.chunk_samples

        self.audio_files = self._scan_files(audio_dir)
        if not self.audio_files:
            raise ValueError(f"No audio files found in: {audio_dir}")

        print(
            f"Dataset configured for {num_chunks} chunks of {samples_per_frame / sampling_rate:.3f}s each. "
            f"Found {len(self.audio_files)} files. Using soundfile backend."
        )

    def _scan_files(self, root: str) -> List[str]:
        extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
        files = []
        for r, _, fs in os.walk(root):
            for f in fs:
                if os.path.splitext(f.lower())[1] in extensions:
                    files.append(os.path.join(r, f))
        return sorted(files)

    def _get_native_sr(self, path: str) -> int:
        """Get native sample rate of the file."""
        info = sf.info(path)
        return info.samplerate

    def _get_duration_samples(self, path: str) -> int:
        """Get total samples at native sample rate."""
        info = sf.info(path)
        return info.frames

    def _load_chunk(self, path: str, offset: int, native_sr: int) -> torch.Tensor:
        """
        Load a chunk at native sample rate.
        Returns tensor of shape [1, total_sequence_samples] at native_sr.
        Note: The caller (tokenizer) should handle resampling if needed.
        """
        target_samples = self.total_sequence_samples

        if native_sr != self.sampling_rate:
            offset = int(offset * native_sr / self.sampling_rate)
            target_samples = int(
                self.total_sequence_samples * native_sr / self.sampling_rate
            )

        try:
            waveform, _ = sf.read(
                path,
                start=offset,
                frames=target_samples,
                dtype="float32",
            )
        except sf.SoundFileError:
            waveform = np.array([])

        if waveform.size == 0:
            waveform = np.zeros(target_samples, dtype=np.float32)

        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]
        else:
            waveform = waveform.T

        if waveform.shape[0] > 1:
            waveform = waveform.mean(axis=0, keepdims=True)

        if waveform.shape[1] < target_samples:
            pad = target_samples - waveform.shape[1]
            waveform = np.pad(waveform, ((0, 0), (0, pad)), mode="constant")
        elif waveform.shape[1] > target_samples:
            waveform = waveform[:, :target_samples]

        return torch.from_numpy(waveform).float()

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

                target_num_samples = num_samples
                if native_sr != self.sampling_rate:
                    target_num_samples = int(
                        num_samples * self.sampling_rate / native_sr
                    )

                max_start = max(0, target_num_samples - self.total_sequence_samples)
                positions = list(range(0, max_start + 1, self.step_size))

                if target_num_samples > self.total_sequence_samples:
                    remaining = target_num_samples - max_start
                    if 0 < remaining < self.total_sequence_samples:
                        positions.append(max_start)

                for start_pos in positions:
                    yield self._load_chunk(file_path, start_pos, native_sr)

            except Exception as e:
                print(f"Warning: Skipping {file_path}: {e}")
                continue


if __name__ == "__main__":
  import time
  dataset = RawAudioDataset(audio_dir="dataset_gen/rotormotor/mp3s_small", num_chunks=8)
  start = time.time()
  for i, sample in enumerate(dataset):
      if i >= 100:
        break
  print(f"100 samples: {time.time() - start:.2f}s")
