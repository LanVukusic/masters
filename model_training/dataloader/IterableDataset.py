# raw_audio_dataset.py
import os
import torch
from torchcodec.decoders import AudioDecoder
from torch.utils.data import IterableDataset
from typing import List, Iterator
import random
from typing import Dict

SAMPLE_FILE = "dataset_gen/free_music/rotormotor/mp3s/001 Guy Contact - Cool Blue Liquid.mp3"

class RawAudioDataset(IterableDataset):
    VALID_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    TARGET_SR = 24000
    FRAME_SIZE = 320  # Fixed: DAC 24kHz stride = 1 token time step

    def __init__(
        self,
        audio_dir: str,
        num_chunks: int = 8,
        shuffle: bool = False,
    ):
        self.audio_dir = audio_dir
        self.num_chunks = num_chunks
        self.shuffle = shuffle
        self.chunk_samples = num_chunks * self.FRAME_SIZE
        
        self.audio_files = self._scan_files(audio_dir)
        if not self.audio_files:
            raise ValueError(f"No audio files found in: {audio_dir}")
            
        print(f"RawAudioDataset: {len(self.audio_files)} files | "
              f"{num_chunks} chunks = {self.chunk_samples/self.TARGET_SR:.3f}s per yield @ {self.TARGET_SR}Hz")

    def _scan_files(self, root: str) -> List[str]:
        files = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if os.path.splitext(fname.lower())[1] in self.VALID_EXTENSIONS:
                    files.append(os.path.join(dirpath, fname))
        return sorted(files)

    def _load_chunk(self, decoder: AudioDecoder, start_sample: int) -> torch.Tensor:
        start_sec = start_sample / self.TARGET_SR
        duration_sec = self.chunk_samples / self.TARGET_SR
        
        segment = decoder.get_samples_played_in_range(
            start_seconds=start_sec, stop_seconds=start_sec + duration_sec
        )
        waveform = segment.data  # [channels, samples]
        print(f"Shape: {waveform.shape}")

        # Mono
        # Convert stereo to mono (torchcodec: dim=0 is channels)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # [1, samples]

        # Add batch dimension: [1, samples] -> [1, 1, samples] for DAC preprocess
        waveform = waveform.unsqueeze(0)  # [1, 1, samples]

        # # Exact length
        # current_len = waveform.shape[1]
        # if current_len < self.chunk_samples:
        #     waveform = torch.nn.functional.pad(waveform, (0, self.chunk_samples - current_len))
        # elif current_len > self.chunk_samples:
        #     waveform = waveform[:, :self.chunk_samples]
        print(f"After mono: {waveform.shape}")
        return waveform  # [1, chunk_samples]

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = max(1, len(self.audio_files) // worker_info.num_workers)
            start_idx = worker_info.id * per_worker
            end_idx = (start_idx + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.audio_files))
            files_to_process = self.audio_files[start_idx:end_idx]
        else:
            files_to_process = self.audio_files

        if self.shuffle:
            random.shuffle(files_to_process)

        for file_path in files_to_process:
            try:
                if(SAMPLE_FILE is not None):
                    decoder = AudioDecoder(SAMPLE_FILE, sample_rate=self.TARGET_SR)
                else:
                    decoder = AudioDecoder(file_path, sample_rate=self.TARGET_SR)
                total_samples = int(decoder.metadata.duration_seconds * self.TARGET_SR)
                
                for start_pos in range(0, total_samples, self.chunk_samples):
                    yield self._load_chunk(decoder, start_pos)

            except Exception as e:
                print(f"Warning: Skipping {file_path}: {e}")
                continue




class TokenizedAudioDataset(IterableDataset):
    def __init__(self, base_dataset: IterableDataset, tokenizer, past_chunks: int, device: str = "cpu"):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.past_chunks = past_chunks
        self.device = device
        self.total_chunks = base_dataset.num_chunks

        if not 0 < past_chunks < self.total_chunks:
            raise ValueError(f"past_chunks must be < {self.total_chunks}")

    def __iter__(self):
        for waveform in self.base_dataset:
            # waveform: [1, samples] → codes: [1, Q, T]
            print("wf shape", waveform.shape)
            codes = self.tokenizer.encode(waveform.to(self.device)).squeeze(0)
            print("codes shape", codes.shape)
            yield {
                "past": codes[:, :self.past_chunks],
                "future": codes[:, self.past_chunks:]
            }

    @staticmethod
    def collate_fn(batch) -> Dict[str, torch.Tensor]:
        return {
            "past": torch.stack([b["past"] for b in batch]),
            "future": torch.stack([b["future"] for b in batch])
        }