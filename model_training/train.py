import sys
import os
import time
import math
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.simple import AudioContinuationTransformer
# from models.conformer.conformer import AudioContinuationConformer

# from model_training.dataloader.raw_dataset import RawAudioDataset
from model_training.dataloader.IterableDataset import (
    RawAudioDataset,
    TokenizedAudioDataset,
)
from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer
from model_training.model_config import (
    MODEL_CONFIG,
    tokens_to_chunks,
)
from utils.visualization import (
    log_audio,
    log_metrics,
    log_visualization,
)


ADVANCED_LOGGING = False
active_model = AudioContinuationTransformer


MODEL_NAME = f"{active_model.__name__}_{time.strftime('%d-%H%M%S')}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if(device == "cuda:0"):
    # clean cache
    torch.cuda.empty_cache()

# Training configuration
config = {
    **MODEL_CONFIG,
    # Training
    "batch_size": 2,
    "learning_rate": 1e-3,
    "num_epochs": 1,
    "num_warmup_steps": 200,
    "gradient_clip": 30.0,
    "training_steps": 1000,
    # Data
    "audio_dir": "dataset_gen/free_music/rotormotor/mp3s_small",
    "tokenizer_type": "DAC",
    # Logging
    "log_audio_every": 50,  # batches
    "log_metrics_every": 10,  # batches
    "log_exposure_every": 100,  # batches
    "validation_every": 100,  # batches
    # Fidelity decay for training
    "use_fidelity_decay": True,
}

# main serves as the multiprocessing guard in python 14.
if __name__ == "__main__":
    global_step = 0

    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{MODEL_NAME}")
    print(f"TensorBoard logs: runs/{MODEL_NAME}")

    model = active_model(config)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=0.01
    )

    # Create a learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=config["training_steps"],
    )

    # Tokenizer
    tokenizer = DACAudioTokenizer(
        num_quantizers=config["n_codebooks"], device=device
    )

    # Dataset (wrapping with tokenizer)
    tokens_needed = config["past_len"] + config["future_len"]
    num_chunks = tokens_to_chunks(tokens_needed)

    raw_dataset = RawAudioDataset(
        audio_dir=config["audio_dir"],
        num_chunks=num_chunks,
        shuffle=True,
    )

    # wrap tokenizer and audio loader in one convenient wrapper
    dataset = TokenizedAudioDataset(
        base_dataset=raw_dataset,
        tokenizer=tokenizer,
        past_chunks=config["past_len"],
        device=device,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        collate_fn=TokenizedAudioDataset.collate_fn,
    )

    # Validation dataset
    val_raw_dataset = RawAudioDataset(
        audio_dir=config["validation_dir"],
        num_chunks=num_chunks,
        shuffle=False,
    )

    val_dataset = TokenizedAudioDataset(
        base_dataset=val_raw_dataset,
        tokenizer=tokenizer,
        past_chunks=config["past_len"],
        device=device,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        collate_fn=TokenizedAudioDataset.collate_fn,
    )


    scaler = torch.amp.GradScaler()  # before training loop

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        if global_step >= config["training_steps"]:
            break

        for batch_idx, batch in enumerate(dataloader):
            iter_start_time = time.time()
            # Batch already tokenized: {"past": [B, K, T_past], "future": [B, K, T_future]}
            past_tokens = batch["past"].to(device, non_blocking=True)
            future_tokens = batch["future"].to(device, non_blocking=False)
            # print("train shapes", past_tokens.shape, future_tokens.shape)

            batch_size_curr, n_cb, total_time = past_tokens.shape

            # Check we have enough tokens
            if future_tokens.shape[-1] < config["future_len"]:
                print(
                    f"Warning: Batch {batch_idx} has insufficient tokens, skipping - {future_tokens.shape[-1]}<{config['future_len']}"
                )
                continue

            future_tokens = future_tokens[:, :, : config["future_len"]].contiguous()

            # =====================================================================
            # 3. Rearrange dimensions for model
            # =====================================================================
            # Model expects: [batch, time, codebooks]
            # Dataset output is: [batch, codebooks, time]
            past_tokens = past_tokens.transpose(1, 2).long()  # [B, T_past, K]
            future_tokens = future_tokens.transpose(1, 2).long()  # [B, T_future, K]

            # =====================================================================
            # 4. Validate token ranges
            # =====================================================================
            max_val = config["vocab_size"] - 1
            if past_tokens.max() > max_val or past_tokens.min() < 0:
                print("Warning: Past tokens out of range")
                sys.exit(1)
            if future_tokens.max() > max_val or future_tokens.min() < 0:
                print("Warning: Future tokens out of range")
                sys.exit(1)

            # =====================================================================
            # 5. Forward pass + Loss + gradient accumulation
            # =====================================================================
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                loss = model.get_training_loss(
                    past_tokens=past_tokens,
                    future_tokens=future_tokens,
                )


            loss_val = loss.item()
            scaler.scale(loss).backward()
            # loss.backward()

            # Gradient clipping
            if config["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["gradient_clip"]
                )

            # optimizer.step()    # update weights
            scaler.step(optimizer)
            scaler.update()
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

                logits = None
                predicted_tokens = None
                with torch.no_grad():
                    logits = model.forward(past_tokens, future_tokens)
                    probs = torch.softmax(logits, dim=-1)
                    predicted_tokens = logits.argmax(dim=-1).flatten().tolist()

                log_metrics(
                    writer=writer,
                    loss=loss_val,
                    lr=scheduler.get_last_lr()[0],
                    global_step=global_step,
                    grad_norm=grad_norm,
                )

                print(
                    f"Epoch {epoch + 1} | step: {global_step:.4f}  | Batch {batch_idx} | Loss: {loss_val:.4f} | Grad: {grad_norm:.3f}"
                )

            # log exposure by generating values autoregresivly twice - expensive
            if global_step % config["log_exposure_every"] == 0 and ADVANCED_LOGGING:
                # 1. Run prediction with Teacher Forcing
                predictions = model.predict(
                    prompt_tokens=past_tokens,
                    true_future_tokens=future_tokens,
                    predict_by_one=True,
                )

                # 2. Calculate Accuracy
                accuracy = (predictions == future_tokens).float().mean()
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

            # log validation
            if global_step % config["validation_every"] == 0 and ADVANCED_LOGGING:
                # Validation at the end of each epoch
                model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        past_tokens = val_batch["past"].to(device, non_blocking=True)
                        future_tokens = val_batch["future"].to(device, non_blocking=False)

                        if future_tokens.shape[-1] < config["future_len"]:
                            continue

                        future_tokens = future_tokens[:, :, :config["future_len"]].contiguous()

                        past_tokens = past_tokens.transpose(1, 2).long()
                        future_tokens = future_tokens.transpose(1, 2).long()

                        loss = model.get_training_loss(
                            past_tokens=past_tokens,
                            future_tokens=future_tokens,
                        )
                        val_loss += loss.item()
                        val_batches += 1
                        if val_batches >= 10:  # Limit to 10 validation batches
                            break

                if val_batches > 0:
                    val_loss /= val_batches
                    writer.add_scalar("Val/Loss", val_loss, global_step)

                model.train()

            # Log audio + spectrogram samples to TensorBoard every N steps
            if global_step % config["log_audio_every"] == 0:
                with torch.no_grad():
                    model.eval()
                    predictions_autoreg = model.predict(
                        past_tokens,
                        temperature=1.0,
                        top_k=100,
                        top_p=0.90,
                        repetition_penalty=1.3,
                    )

                    gt_waveform, pred_waveform = log_audio(
                        writer=writer,
                        tokenizer=tokenizer,
                        future_tokens=future_tokens,
                        predictions_autoreg=predictions_autoreg,
                        global_step=global_step,
                        future_len=config["future_len"],
                        audio_log_level=config["n_codebooks"],
                    )

                    log_visualization(
                        writer=writer,
                        gt_waveform=gt_waveform,
                        pred_waveform=pred_waveform,
                        global_step=global_step,
                    )

                    del predictions_autoreg, gt_waveform, pred_waveform
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    model.train()

            iter_time = time.time() - iter_start_time
            time_per_sample = iter_time / config["batch_size"]
            # print(f"{iter_time:.1f}s / iter - {time_per_sample:.2f}s / sample")

            global_step += 1

       
    writer.close()
    # Final save
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        f"checkpoints/{MODEL_NAME}_final.pt",
    )
    print(f"Final model saved: checkpoints/{MODEL_NAME}_final.pt")
