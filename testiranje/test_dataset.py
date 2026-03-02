import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from transformers import MimiModel, AutoFeatureExtractor

audio_dir = "."


# Mimi Encode Transform using Hugging Face Transformers
class MimiEncodeTransform(object):
    def __init__(self, model_name="kyutai/mimi", device="cpu"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = MimiModel.from_pretrained(model_name).to(device)
        self.device = device

    def __call__(self, audio_windows):
        # audio_windows is expected to be (num_windows, window_size)
        encoded_codes = []
        with torch.no_grad():
            for window in audio_windows:
                # Pre-process the inputs
                # The feature_extractor can often accept a torch.Tensor directly if it's 1D
                inputs = self.feature_extractor(
                    raw_audio=window,  # Pass torch.Tensor directly
                    sampling_rate=self.feature_extractor.sampling_rate,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Encode the audio inputs
                # The model.encode method expects input_values of shape (batch_size, sequence_length)
                # and returns audio_codes of shape (batch_size, num_quantizers, sequence_length_codes)
                encoder_outputs = self.model.encode(
                    inputs["input_values"], num_quantizers=32
                )  # Using 32 codebooks for Mimi model
                encoded_codes.append(
                    encoder_outputs.audio_codes.squeeze(0)
                )  # Remove batch dim, resyult (K, T_codes)
        return torch.stack(encoded_codes)  # Resulting shape: (num_windows, K, T_codes)


# Parameters for the dataset
sampling_rate = 16000
number_of_frames = 40  # Corresponds to approx 5 seconds at 16kHz with FRAME_SIZE=1920
window_size_multiplier = 4  # 4 * FRAME_SIZE

# Define a transform (using MimiEncodeTransform)
mimi_transform = MimiEncodeTransform(device="cpu")

# Create an instance of the AudioDataset with the MimiEncodeTransform
dataset = AudioDataset(
    audio_dir=audio_dir,
    sampling_rate=sampling_rate,
    number_of_frames=number_of_frames,
    window_size_multiplier=window_size_multiplier,
    transform=mimi_transform,
)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"Dataset created with {len(dataset)} audio files.")
print(
    f"Sampling rate: {sampling_rate}, Number of frames: {number_of_frames}, Window size multiplier: {window_size_multiplier} (actual window size: {dataset.window_size})"
)

# Iterate through a few batches to test
for i, batch in enumerate(dataloader):
    print(f"\nBatch {i + 1} shape: {batch.shape}")
    # Expected shape: (batch_size, num_windows, K, T_codes)
    # K is number of codebooks (32 for mimi)
    # T_codes is the number of codebook tokens per window.
    # For window_size = 7680 and FRAME_SIZE = 1920, T_codes = window_size / FRAME_SIZE = 7680 / 1920 = 4
    # So, batch shape should be (batch_size, 10, 32, 4)

    if i >= 1:  # Test 2 batches
        break
