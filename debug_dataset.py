import sys
import os
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data import DataLoader
from model_training.dataloader.audio_tokenized_dataset import AudioTokenizedDataset
from model_training.tokenizer.audio_tokenizer import AudioTokenizer

# Configuration
config = {
    "history_length": 4,
    "future_frames": 1,
    "rvq_levels": 8,
    "batch_size": 2,
    "audio_dir": "dataset_gen/rotormotor/mp3s",
}

# Create tokenizer
tokenizer = AudioTokenizer(
    num_quantizers=config["rvq_levels"],
    device="cpu",
)

# Create dataset
dataset = AudioTokenizedDataset(
    audio_dir=config["audio_dir"],
    tokenizer=tokenizer,
    num_chunks=config["history_length"] + config["future_frames"],
    rvq_depth=config["rvq_levels"],
    chunk_duration=2.0,
    max_samples_per_file=5,
)

print(f"Dataset created successfully!")
print(f"Number of audio files: {len(dataset.audio_files)}")
print(f"Estimated dataset length: {len(dataset)}")

# Test a few samples to see the actual token ranges
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

print("\nTesting token ranges in dataset:")
for i, batch in enumerate(dataloader):
    print(f"Batch {i + 1} shape: {batch.shape}")
    print(f"Batch min: {batch.min()}, max: {batch.max()}")
    print(f"Batch unique values range: {len(torch.unique(batch))}")
    
    # Check per level
    for level in range(batch.shape[1]):
        level_tokens = batch[:, level, :]
        print(f"  Level {level}: min={level_tokens.min()}, max={level_tokens.max()}")
    
    if i >= 2:  # Test a few batches
        break
