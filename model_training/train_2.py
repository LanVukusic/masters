import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_testing.model import TestModel  # Updated import
from model_training.dataloader.raw_dataset import RawAudioDataset
from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer
from model_training.tokenizer.mimi_audio_tokenizer import MimiAudioTokenizer
import time

MODEL_NAME = f"audio_continuation_{time.strftime('DD.HH:mm')}"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# Configuration
config = {
    "history_length": 10,
    "future_length": 4,
    "rvq_levels": 16,
    "embedding_dim": 256,
    "batch_size": 6,
    "codebook_size": 1024,  # Updated to match Mimi tokenizer range (0-2047) or DAC 1024
    "learning_rate": 3e-4,
    "num_epochs": 1000,
    "audio_dir": "dataset_gen/rotormotor/mp3s_small",
    "tokenizer_type": "DAC",  # or "MIMI"
    "device": device,
}


# Create dataset and dataloader
dataset = RawAudioDataset(
    audio_dir=config["audio_dir"],
    tokenizer_type=config["tokenizer_type"],
    num_chunks=config["history_length"]
    + config["future_length"],  # Total sequence length
    cache_size=3,  # Reduced cache size to prevent memory issues on GPU server
)

dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=1,  # Reduced from 8 to 1 to prevent memory issues on GPU server
    pin_memory=True,  # Enable to make memory transfer faster
    persistent_workers=True,  # keep workers alive between epochs - expensive init
)

print("Dataset created successfully!")
print(f"Number of audio files found: {len(dataset.audio_files)}")
print(f"Estimated dataset length: {len(dataset)}")
print(f"Dataloader batch size: {config['batch_size']}")

if config["tokenizer_type"] == "DAC":
    tokenizer = DACAudioTokenizer(num_quantizers=config["rvq_levels"], device=device)
else:
    tokenizer = MimiAudioTokenizer(num_quantizers=config["rvq_levels"], device=device)


# Initialize model AFTER dataset creation to ensure correct codebook_size
model = TestModel(  # Updated model class name
    history_length=config["history_length"],
    future_length=config["future_length"],
    rvq_levels=config["rvq_levels"],
    embedding_dim=config["embedding_dim"],
    codebook_size=config["codebook_size"],
    trans_layers=2,  # Added required parameters
    num_heads=8,
    conv_channels=128,
)
model.to(device)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/{MODEL_NAME}")

# Training loop with actual data from dataloader
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])


def compute_cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss for multi-level RVQ prediction.

    Args:
        logits: [B, future_length, rvq_levels, codebook_size] - model output
        targets: [B, future_length, rvq_levels] - ground truth tokens

    Returns:
        scalar loss value
    """
    # Reshape for cross-entropy computation
    B, future_length, rvq_levels, codebook_size = logits.shape
    logits_flat = logits.reshape(
        -1, codebook_size
    )  # [B * future_length * rvq_levels, codebook_size]
    targets_flat = targets.reshape(-1)  # [B * future_length * rvq_levels]

    # Compute cross-entropy loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits_flat, targets_flat)

    return loss


print("Starting training...")
for epoch in range(config["num_epochs"]):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for batch_idx, raw_audio_batch in enumerate(dataloader):
        raw_audio_gpu = raw_audio_batch.to(
            device, non_blocking=True
        )  # non_blocking=True helps overlap transfer/computation
        # Tokenize ON THE GPU - this is where the heavy lifting happens efficiently
        with (
            torch.no_grad()
        ):  # Tokenization doesn't require gradients during data loading
            batch = tokenizer.encode_from_waveform(
                raw_audio_gpu, tokenizer.sampling_rate
            )

        # batch shape: [batch_size, rvq_levels, time_steps]
        batch_size_current, rvq_levels, total_time_steps = batch.shape

        # Split the batch into historical and future parts
        # Assuming each chunk represents one time step
        frames_per_chunk = (1920 * tokenizer.sampling_rate) // dataset.samples_per_frame

        # We need to split the temporal dimension appropriately
        total_chunks_needed = config["history_length"] + config["future_length"]

        # Calculate how many time steps correspond to our needed chunks
        time_steps_per_chunk = total_time_steps // total_chunks_needed

        if time_steps_per_chunk < 1:
            print(
                f"Warning: Not enough time steps in batch. Skipping batch {batch_idx}"
            )
            continue

        # Extract historical and future RVQ tokens
        historical_end = config["history_length"] * time_steps_per_chunk
        future_start = historical_end
        future_end = future_start + config["future_length"] * time_steps_per_chunk

        historical_rvq = batch[
            :, :, :historical_end
        ]  # [batch_size, rvq_levels, history_time_steps]
        future_rvq = batch[
            :, :, future_start:future_end
        ]  # [batch_size, rvq_levels, future_time_steps]

        # Rearrange to match model expectations: [batch_size, time_steps, rvq_levels]
        historical_rvq = historical_rvq.transpose(
            1, 2
        )  # [batch_size, history_time_steps, rvq_levels]
        future_rvq = future_rvq.transpose(
            1, 2
        )  # [batch_size, future_time_steps, rvq_levels]

        # Validate token ranges to prevent index out of bounds
        max_token_value = config["codebook_size"] - 1
        if historical_rvq.max() > max_token_value or historical_rvq.min() < 0:
            print(
                f"Warning: Historical RVQ tokens out of range! Max: {historical_rvq.max()}, Min: {historical_rvq.min()}"
            )
            # Clamp values to valid range
            historical_rvq = torch.clamp(historical_rvq, 0, max_token_value)

        if future_rvq.max() > max_token_value or future_rvq.min() < 0:
            print(
                f"Warning: Future RVQ tokens out of range! Max: {future_rvq.max()}, Min: {future_rvq.min()}"
            )
            # Clamp values to valid range
            future_rvq = torch.clamp(future_rvq, 0, max_token_value)

        # Convert to long type (required for embedding lookup)
        historical_rvq = historical_rvq.long()
        future_rvq = future_rvq.long()

        # Apply adaptive pooling to match expected temporal dimensions
        # The model expects historical_rvq to have temporal_length = history_length
        # and future_rvq to have temporal_length = future_length
        if historical_rvq.size(1) != config["history_length"]:
            # Pool historical_rvq to match history_length
            historical_rvq_float = historical_rvq.float()
            pooled_historical = (
                torch.nn.functional.adaptive_avg_pool1d(
                    historical_rvq_float.transpose(
                        1, 2
                    ),  # -> [B, rvq_levels, temporal_length]
                    config["history_length"],
                )
                .transpose(1, 2)
                .long()
            )  # -> [B, history_length, rvq_levels]
            historical_rvq = pooled_historical

        if future_rvq.size(1) != config["future_length"]:
            # Pool future_rvq to match future_length
            future_rvq_float = future_rvq.float()
            pooled_future = (
                torch.nn.functional.adaptive_avg_pool1d(
                    future_rvq_float.transpose(
                        1, 2
                    ),  # -> [B, rvq_levels, temporal_length]
                    config["future_length"],
                )
                .transpose(1, 2)
                .long()
            )  # -> [B, future_length, rvq_levels]
            future_rvq = pooled_future

        optimizer.zero_grad()

        # Forward pass - model now returns logits instead of a dictionary
        logits = model(historical_rvq)  # [B, future_length, rvq_levels, codebook_size]

        # Compute loss
        loss = compute_cross_entropy_loss(logits, future_rvq)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

        # Log batch-level metrics every 10 batches
        if batch_idx % 10 == 0:
            writer.add_scalar(
                "Batch/Loss", loss.item(), epoch * len(dataloader) + batch_idx
            )
            writer.add_scalar(
                "Batch/Learning_Rate",
                optimizer.param_groups[0]["lr"],
                epoch * len(dataloader) + batch_idx,
            )

            # Log gradient norm
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm**0.5
            writer.add_scalar(
                "Batch/Gradient_Norm",
                total_grad_norm,
                epoch * len(dataloader) + batch_idx,
            )

        # Print progress for first few batches in each epoch
        print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

    if batch_count > 0:
        avg_epoch_loss = epoch_loss / batch_count
        print(
            f"Epoch {epoch + 1}/{config['num_epochs']}, Average Loss: {avg_epoch_loss:.4f}"
        )

        # Log epoch-level metrics
        writer.add_scalar("Epoch/Average_Loss", avg_epoch_loss, epoch)
        writer.add_scalar("Epoch/Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        # Log gradient norm at epoch level
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm**0.5
        writer.add_scalar("Epoch/Gradient_Norm", total_grad_norm, epoch)
    else:
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, No valid batches processed")


# Close TensorBoard writer
writer.close()

# Save model
torch.save(model.state_dict(), f"{MODEL_NAME}.pth")
print("Model saved successfully!")
