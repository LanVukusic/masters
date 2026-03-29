import sys
import os
import time
import math
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from simpleModel.simple_v2 import AudioContinuationTransformer
from model_training.dataloader.raw_dataset import RawAudioDataset
from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer
from model_training.model_config import MODEL_CONFIG, TOKEN_RATE
from utils.logging import (
    log_audio_samples,
    log_training_metrics,
    log_dj_waveform,
)

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = f"audio_AUTOREG_{time.strftime('%d-%H%M')}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training configuration
config = {
    **MODEL_CONFIG,
    # Training
    "batch_size": 1,
    "learning_rate": 1e-2,
    "num_epochs": 20,
    "num_warmup_epochs": 2,
    "gradient_clip": 0.0,
    # Data
    "audio_dir": "dataset_gen/rotormotor/mp3s_small",
    # "audio_dir": "dataset_gen/rotormotor/mp3s",
    "tokenizer_type": "DAC",
    # Logging
    "log_audio_every": 150,  # batches
    "log_metrics_every": 10,  # batches
    "log_exposure_every": 100,  # batches
    # Fidelity decay for training
    "use_fidelity_decay": True,
    # Progressive teacher forcing
    "use_progressive_tf": True,
    "tf_warmup_steps": 500,  # Steps to go from full TF to no TF
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

# Limit dataset size for faster training (remove this for full training)
max_samples = 500
if len(dataset) > max_samples:
    dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))
    print(f"Limited dataset to {max_samples} samples for faster training")

dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,  # Shuffle for training
    num_workers=0,  # Reduce for stability with on-the-fly tokenization
    pin_memory=True,
)

print(f"Dataset: {len(dataset)} chunks")


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

# Create a learning rate scheduler with warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
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
            raw_audio_gpu = raw_audio_gpu.squeeze(1)
            tokens_list = []
            for i in range(raw_audio_gpu.shape[0]):
                single_audio = raw_audio_gpu[i : i + 1]
                codes = tokenizer.encode_from_waveform(
                    single_audio, original_sampling_rate=tokenizer.sampling_rate
                )
                tokens_list.append(codes)
            tokens = torch.cat(tokens_list, dim=0)

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
        # 5. Forward pass + Loss + gradient accumulation
        # =====================================================================
        optimizer.zero_grad()

        # Calculate teacher forcing ratio for progressive teacher forcing
        if config.get("use_progressive_tf", False):
            tf_warmup_steps = config.get("tf_warmup_steps", 500)
            tf_ratio = max(0.0, 1.0 - (global_step / tf_warmup_steps))
        else:
            tf_ratio = 1.0  # Full teacher forcing

        # Use progressive teacher forcing or standard loss
        if config.get("use_progressive_tf", False) and tf_ratio < 1.0:
            loss = model.get_progressive_teacher_forcing_loss(
                past_tokens=past_tokens,
                future_tokens=future_tokens,
                teacher_forcing_ratio=tf_ratio,
            )
            loss_val = loss.item()
        else:
            # Standard teacher forcing loss
            loss = model.get_training_loss(
                past_tokens=past_tokens,
                future_tokens=future_tokens,
            )
            loss_val = loss.item()

        loss.backward()

        # Gradient clipping
        if config["gradient_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])

        optimizer.step()  # update weights
        scheduler.step()  # update lr scheduler

        # =====================================================================
        # 7. Logging
        # =====================================================================
        epoch_loss += loss_val
        valid_batches += 1

        # Log metrics every N batches
        if global_step % config["log_metrics_every"] == 0:
            grad_norm = math.sqrt(
                sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                )
            )

            tf_ratio_val = tf_ratio if config.get("use_progressive_tf", False) else None
            log_training_metrics(
                writer=writer,
                loss=loss_val,
                lr=scheduler.get_last_lr()[0],
                global_step=global_step,
                grad_norm=grad_norm,
                teacher_forcing_ratio=tf_ratio_val,
            )

            print(
                f"Epoch {epoch + 1} | Batch {batch_idx} | Loss: {loss_val:.4f} | Grad: {grad_norm:.3f}"
            )

        # log exposure by generating values autoregresivly twice - expensive
        if global_step % config["log_exposure_every"] == 0:
            # 1. Run prediction with Teacher Forcing
            predictions = model.predict(
                prompt_tokens=past_tokens,
                true_future_tokens=future_tokens,
                predict_by_one=True,
            )

            # 2. Calculate Accuracy
            # predictions shape: (Batch, T_future, N_Codebooks)
            # future_tokens shape: (Batch, T_future, N_Codebooks)

            accuracy = (predictions == future_tokens).float().mean()
            # print(f"Next Token Prediction Accuracy: {accuracy.item():.4f}")
            writer.add_scalar("Train/Accuracy", accuracy, global_step)

            # 3. Compare with Standard Inference (to measure Exposure Bias)
            predictions_free = model.predict(
                prompt_tokens=past_tokens,
                temperature=1.0,
                top_k=200,
                top_p=0.95,
                repetition_penalty=1.1,
                predict_by_one=False,
            )
            accuracy_free = (predictions_free == future_tokens).float().mean()

            writer.add_scalar(
                "Train/ExposureBias",
                accuracy.item() - accuracy_free.item(),
                global_step,
            )

        # Log audio samples periodically
        if global_step % config["log_audio_every"] == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                f"checkpoints/{MODEL_NAME}_mid.pt",
            )
            print(f"mid model saved: checkpoints/{MODEL_NAME}_mid.pt")

            with torch.no_grad():
                model.eval()

                # 1. One-token prediction: predict every token using ground truth context
                predictions_one_token = model.predict(
                    past_tokens,
                    temperature=1.0,
                    top_k=200,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    predict_by_one=True,
                    true_future_tokens=future_tokens,
                )

                # 2. Full autoregressive: model uses its own predictions as context
                predictions_autoreg = model.predict(
                    past_tokens,
                    temperature=1.0,
                    top_k=200,
                    top_p=0.95,
                    repetition_penalty=1.1,
                )

                # Log audio samples
                gt_waveform, pred_waveform = log_audio_samples(
                    writer=writer,
                    tokenizer=tokenizer,
                    past_tokens=past_tokens,
                    future_tokens=future_tokens,
                    predictions_one_token=predictions_one_token,
                    predictions_autoreg=predictions_autoreg,
                    global_step=global_step,
                    future_len=config["future_len"],
                    audio_log_level=4,
                )

                # Log DJ waveform visualization
                if gt_waveform is not None and pred_waveform is not None:
                    log_dj_waveform(
                        writer=writer,
                        gt_waveform=gt_waveform,
                        pred_waveform=pred_waveform,
                        global_step=global_step,
                    )

                # Explicit cleanup to prevent memory leaks
                del predictions_one_token, predictions_autoreg
                del gt_waveform, pred_waveform
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                model.train()

        global_step += 1

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
