import os
import torch
import torchaudio
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Iterator

class RawAudioDataset(IterableDataset):
    """
    Optimized IterableDataset for audio. 
    - No upfront metadata loading (fast init).
    - Lazy probing via torchaudio.info.
    - Worker-aware file splitting.
    - API compatible constructor (but iteration only, no indexing).
    """

    def __init__(
        self,
        audio_dir: str,
        num_chunks: int = 8,
        sampling_rate: int = 24000,
        samples_per_frame: int = 320,
        cache_size: int = 3,  # Kept for API compatibility, logic simplified
    ):
        self.audio_dir = audio_dir
        self.num_chunks = num_chunks
        self.sampling_rate = sampling_rate
        self.samples_per_frame = samples_per_frame
        
        self.chunk_samples = samples_per_frame
        self.total_sequence_samples = self.chunk_samples * num_chunks
        self.step_size = self.chunk_samples

        # Scan files only (fast, no probing)
        self.audio_files = self._scan_files(audio_dir)
        if not self.audio_files:
            raise ValueError(f"No audio files found in: {audio_dir}")

    def _scan_files(self, root: str) -> List[str]:
        extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
        files = []
        for r, _, fs in os.walk(root):
            for f in fs:
                if os.path.splitext(f.lower())[1] in extensions:
                    files.append(os.path.join(r, f))
        return files

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = get_worker_info()
        
        # Split files among workers
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            per_worker = max(1, len(self.audio_files) // num_workers)
            start = worker_id * per_worker
            # Last worker takes remainder
            end = start + per_worker if worker_id < num_workers - 1 else len(self.audio_files)
            worker_files = self.audio_files[start:end]
        else:
            worker_files = self.audio_files

        # Shuffle file order per worker per epoch
        # Note: Use a generator-friendly shuffle if deterministic behavior is needed
        import random
        worker_files = list(worker_files) 
        random.shuffle(worker_files)

        for file_path in worker_files:
            try:
                # Lazy probe
                info = torchaudio.info(file_path)
                num_samples = info.num_frames
                
                # Calculate valid starts
                max_start = max(0, num_samples - self.total_sequence_samples)
                positions = range(0, max_start + 1, self.step_size)
                
                # Handle trailing edge if needed
                if num_samples > self.total_sequence_samples:
                    remaining = num_samples - max_start
                    if 0 < remaining < self.total_sequence_samples:
                        positions = list(positions) + [max_start]

                for start_pos in positions:
                    yield self._load_chunk(file_path, start_pos)
                    
            except Exception as e:
                print(f"Warning: Skipping {file_path}: {e}")
                continue

    def _load_chunk(self, path: str, offset: int) -> torch.Tensor:
        # Load only required frames (memory efficient)
        # frame_offset is in frames, not samples. Assuming 1 channel logic for offset calc
        # torchaudio.load offset is in frames. 
        # Note: torchaudio.load frame_offset expects frames, not samples. 
        # For mono 1 frame = 1 sample. For multi-channel, frames = samples.
        
        try:
            waveform, sr = torchaudio.load(
                path, 
                frame_offset=offset, 
                num_frames=self.total_sequence_samples
            )
        except Exception:
            # Fallback if seek not supported (load all then slice)
            waveform, sr = torchaudio.load(path)
            waveform = waveform[:, offset:offset + self.total_sequence_samples]

        # Process
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

        # Pad if necessary (e.g., end of file)
        if waveform.shape[1] < self.total_sequence_samples:
            pad = self.total_sequence_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform

# Usage Note:
# IterableDataset does not support len() or indexing (dataset[0]).
# Iterate directly: for batch in dataset: ...