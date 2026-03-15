import sys
import os
import time
import math
from pathlib import Path
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from narTransformer.narTrans import AudioForecaster
from simpleModel.simple import AudioContinuationTransformer
from model_training.dataloader.raw_dataset import RawAudioDataset
from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = f"audio_NAR_{time.strftime('%d-%H%M')}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DAC token rate: ~86 Hz at 24kHz, or ~50 Hz if downsampled
# Adjust based on your tokenizer's actual output rate
TOKEN_RATE = 100  # tokens per second

# Training configuration
config = {
    # Sequence lengths (in tokens, not seconds)
    "past_len": int(2 * TOKEN_RATE),  # 5 seconds of context
    "future_len": int(2 * TOKEN_RATE),  # 10 seconds to predict
    # Model architecture
    "vocab_size": 1024,  # DAC codebook size
    "n_codebooks": 8,  # DAC RVQ layers
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 4,
    "d_ff": 1024,
    "dropout": 0.15,
    # Training
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_epochs": 100,
    "num_warmup_epochs": 2,
    "gradient_clip": 1.0,
    # Data
    "audio_dir": "dataset_gen/rotormotor/mp3s_small",
    "tokenizer_type": "DAC",
    # Logging
    "log_audio_every": 150,  # batches
    "log_metrics_every": 10,  # batches
    # Fidelity decay for training
    "use_fidelity_decay": True,
}

# =============================================================================
# Initialize Components
# =============================================================================

# Tokenizer
tokenizer = DACAudioTokenizer(num_quantizers=config["n_codebooks"], device=device)

# Dataset
dataset = RawAudioDataset(
    audio_dir=config["audio_dir"],
    num_chunks=config["past_len"] + config["future_len"],  # Total tokens needed
    cache_size=3,
)

dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,  # Shuffle for training
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

print(f"Dataset: {len(dataset.audio_files)} files, {len(dataset)} chunks")

# Model - initialize with config values
# model = AudioForecaster(
#     vocab_size=config["vocab_size"],
#     d_model=config["d_model"],
#     n_heads=config["n_heads"],
#     n_layers=config["n_layers"],
#     d_ff=config["d_ff"],
#     n_codebooks=config["n_codebooks"],
#     past_len=config["past_len"],
#     future_len=config["future_len"],
#     dropout=config["dropout"],
#     device=device,  # Pass device so mask buffer is created on correct device
# )

model = AudioContinuationTransformer(config)
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config["learning_rate"], weight_decay=0.01
)

total_batches = len(dataloader)
num_training_steps = total_batches * config["num_epochs"]
num_warmup_steps = total_batches * config["num_warmup_epochs"]

print(f"Total training steps: {num_training_steps}")
print(f"Total warmup steps: {num_warmup_steps}")

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)


# Define warmup and total steps
# Let's say we want to warm up for the first 5% of the total training steps
num_warmup_steps = int(0.05 * (len(dataloader) * config["num_epochs"]))
num_training_steps = len(dataloader) * config["num_epochs"]

# Create a scheduler
# A cosine scheduler with warmup is a very robust choice
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_training_steps - num_warmup_steps,
    eta_min=1e-6,  # Don't let the LR go to zero
)


# TensorBoard
writer = SummaryWriter(log_dir=f"runs/{MODEL_NAME}")
print(f"TensorBoard logs: runs/{MODEL_NAME}")

# =============================================================================
# Training Loop
# =============================================================================

print("Starting training...")
global_step = 0

for epoch in range(config["num_epochs"]):
    model.train()
    epoch_loss = 0.0
    valid_batches = 0

    for batch_idx, raw_audio_batch in enumerate(dataloader):
        # Move raw audio to device
        raw_audio_gpu = raw_audio_batch.to(device, non_blocking=True)

        # =====================================================================
        # 1. Tokenize audio (no gradients needed)
        # =====================================================================
        with torch.no_grad():
            # encode_from_waveform should return: (batch, codebooks, time)
            tokens = tokenizer.encode_from_waveform(
                raw_audio_gpu, original_sampling_rate=tokenizer.sampling_rate
            )

        # tokens shape: [batch, n_codebooks, total_time]
        batch_size_curr, n_cb, total_time = tokens.shape

        # =====================================================================
        # 2. Split into past (input) and future (target)
        # =====================================================================
        past_tokens = tokens[:, :, : config["past_len"]]  # [B, K, T_past]
        future_tokens = tokens[
            :, :, config["past_len"] : config["past_len"] + config["future_len"]
        ]  # [B, K, T_future]

        # Check we have enough tokens
        if future_tokens.shape[-1] < config["future_len"]:
            print(f"Warning: Batch {batch_idx} has insufficient tokens, skipping")
            continue

        # =====================================================================
        # 3. Rearrange dimensions for model
        # =====================================================================
        # Model expects: [batch, time, codebooks]
        # Tokens are: [batch, codebooks, time]
        past_tokens = past_tokens.transpose(1, 2).long()  # [B, T_past, K]
        future_tokens = future_tokens.transpose(1, 2).long()  # [B, T_future, K]

        # =====================================================================
        # 4. Validate token ranges
        # =====================================================================
        max_val = config["vocab_size"] - 1
        if past_tokens.max() > max_val or past_tokens.min() < 0:
            print("Warning: Past tokens out of range")
            os.exit(1)
            # past_tokens = torch.clamp(past_tokens, 0, max_val)
        if future_tokens.max() > max_val or future_tokens.min() < 0:
            print("Warning: Future tokens out of range")
            os.exit(1)
            # future_tokens = torch.clamp(future_tokens, 0, max_val)

        # =====================================================================
        # 5. Forward pass + Loss
        # =====================================================================
        optimizer.zero_grad()

        # Use the model's built-in loss function
        loss = model.get_training_loss(
            past_tokens=past_tokens,
            future_tokens=future_tokens,
        )

        # =====================================================================
        # 6. Backward pass
        # =====================================================================
        loss.backward()

        # Gradient clipping
        if config["gradient_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])

        optimizer.step()

        # =====================================================================
        # 7. Logging
        # =====================================================================
        epoch_loss += loss.item()
        valid_batches += 1

        # Log metrics every N batches
        if batch_idx % config["log_metrics_every"] == 0:
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/LR", scheduler.get_lr(), global_step)

            # Gradient norm
            grad_norm = torch.sqrt(
                sum(
                    torch.square(p.grad.norm().item())
                    for p in model.parameters()
                    if p.grad is not None
                )
            )
            writer.add_scalar("Train/GradNorm", grad_norm, global_step)

            print(
                f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss.item():.4f} | Grad: {grad_norm:.3f}"
            )

        # Log audio samples periodically
        if batch_idx % config["log_audio_every"] == 0:
            with torch.no_grad():
                model.eval()

                # Generate predictions
                predictions = model.predict(
                    past_tokens, temperature=0.9, top_k=50
                )  # [B, T_future, K]

                # Decode to waveform (use only high-fidelity codebooks for first 2s)
                # For thesis: you might want to decode only codebooks 0-3 for immediate audio
                high_fidelity_tokens = predictions[
                    :, : int(config["future_len"]), :4
                ]  # first 4 codebooks

                if high_fidelity_tokens.numel() > 0:
                    try:
                        waveform = tokenizer.decode_to_waveform(
                            high_fidelity_tokens.transpose(1, 2)  # [B, K, T]
                        )
                        writer.add_audio(
                            "Audio/Prediction",
                            waveform[0].cpu(),
                            global_step,
                            sample_rate=tokenizer.sampling_rate,
                        )

                        # Also log ground truth for comparison
                        gt_tokens = future_tokens[:, : int(config["future_len"]), :4]
                        gt_waveform = tokenizer.decode_to_waveform(
                            gt_tokens.transpose(1, 2)
                        )
                        writer.add_audio(
                            "Audio/GroundTruth",
                            gt_waveform[0].cpu(),
                            global_step,
                            sample_rate=tokenizer.sampling_rate,
                        )
                    except Exception as e:
                        print(f"Warning: Could not decode audio for logging: {e}")

                model.train()

        global_step += 1

    # # =====================================================================
    # # Epoch Summary
    # # =====================================================================
    # if valid_batches > 0:
    #     avg_loss = epoch_loss / valid_batches
    #     print(f"Epoch {epoch + 1}/{config['num_epochs']} | Avg Loss: {avg_loss:.4f}")

    #     writer.add_scalar("Epoch/AvgLoss", avg_loss, epoch)

    #     # Save checkpoint every 10 epochs
    #     if (epoch + 1) % 10 == 0:
    #         checkpoint = {
    #             "epoch": epoch,
    #             "model_state_dict": model.state_dict(),
    #             "optimizer_state_dict": optimizer.state_dict(),
    #             "loss": avg_loss,
    #             "config": config,
    #         }
    #         save_path = f"checkpoints/{MODEL_NAME}_epoch{epoch + 1}.pt"
    #         Path("checkpoints").mkdir(exist_ok=True)
    #         torch.save(checkpoint, save_path)
    #         print(f"Checkpoint saved: {save_path}")
    # else:
    #     print(f"Epoch {epoch + 1}: No valid batches processed")

# =============================================================================
# Cleanup
# =============================================================================
writer.close()

# # Final save
# torch.save(
#     {
#         "model_state_dict": model.state_dict(),
#         "config": config,
#     },
#     f"checkpoints/{MODEL_NAME}_final.pt",
# )
# print(f"Final model saved: checkpoints/{MODEL_NAME}_final.pt")
