import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    FRAME_SIZE = 1920  # Hardcoded frame size

    def __init__(
        self,
        audio_dir,
        sampling_rate=16000,
        number_of_frames=42,  # Default to 42 frames (approx 5 seconds at 16kHz)
        window_size_multiplier=1,  # Default window_size to 1 * FRAME_SIZE
        transform=None,
    ):
        self.audio_dir = audio_dir
        self.sampling_rate = sampling_rate
        self.number_of_frames = number_of_frames
        self.window_size = window_size_multiplier * self.FRAME_SIZE
        self.transform = transform

        self.audio_files = []
        for root, _, files in os.walk(audio_dir):
            for file in files:
                if file.endswith((".mp3", ".wav", ".flac", ".ogg")):
                    self.audio_files.append(os.path.join(root, file))

        if not self.audio_files:
            raise ValueError(f"No audio files found in the directory: {audio_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform_orig, original_sampling_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform_orig.shape[0] > 1:
            waveform_orig = torch.mean(waveform_orig, dim=0, keepdim=True)

        # Resample if necessary
        if original_sampling_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sampling_rate, new_freq=self.sampling_rate
            )
            waveform_orig = resampler(waveform_orig)

        # Determine the total number of samples for the desired number_of_frames
        target_num_samples = self.number_of_frames * self.FRAME_SIZE

        # The target_num_samples is already a multiple of FRAME_SIZE by definition,
        # so no additional check for frame_size is needed here.
        # However, we still need to ensure it's a multiple of window_size for unfolding later.

        # If the original waveform is shorter than target_num_samples, pad it
        if waveform_orig.shape[1] < target_num_samples:
            padding_needed = target_num_samples - waveform_orig.shape[1]
            waveform_orig = torch.nn.functional.pad(waveform_orig, (0, padding_needed))
            start_sample = 0  # If padded, start from the beginning
        else:
            # Randomly select a chunk
            max_start_sample = waveform_orig.shape[1] - target_num_samples
            start_sample = random.randint(0, max_start_sample)

        chunk = waveform_orig[:, start_sample : start_sample + target_num_samples]

        # Split into windows
        # Ensure chunk length is a multiple of window_size for unfolding
        # This is important if number_of_frames * FRAME_SIZE is not a multiple of window_size
        if target_num_samples % self.window_size != 0:
            padding_needed = self.window_size - (target_num_samples % self.window_size)
            chunk = torch.nn.functional.pad(chunk, (0, padding_needed))

        # Use unfold to create windows
        # The output shape will be (1, num_windows, window_size) if input is (1, total_samples)
        windows = chunk.unfold(
            dimension=-1, size=self.window_size, step=self.window_size
        )

        # Use unfold to create windows
        # The output shape will be (1, num_windows, window_size) if input is (1, total_samples)
        windows = chunk.unfold(
            dimension=-1, size=self.window_size, step=self.window_size
        )

        # Squeeze the first dimension if it's 1 (from mono channel)
        # Resulting shape: (num_windows, window_size)
        sample = windows.squeeze(0)

        if self.transform:
            sample = self.transform(sample)

        return sample
