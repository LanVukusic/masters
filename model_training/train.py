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
from model_training.dataloader.IterableDataset import RawAudioDataset
from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer
from model_training.model_config import (
    MODEL_CONFIG,
    tokens_to_chunks,
)
from utils.visualization import (
    log_audio,
    log_codebook_metrics,
    log_generation_stats,
    log_metrics,
    log_visualization,
)


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
    "batch_size": 24,
    "learning_rate": 1e-3,         # safe with AdamW on small transformer; bump to 2e-3 only if stable
    "num_epochs": 1,
    "num_warmup_steps": 500,       # ~10% of training_steps, standard
    "gradient_clip": 1.0,
    "training_steps": 5000,
    # Data
    "audio_dir": "dataset_gen/free_music/rotormotor/mp3s",
    "validation_dir": "dataset_gen/free_music/rotormotor/validation",
    "tokenizer_type": "DAC",
    # Logging
    "log_audio_every": 150,  # batches
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
        overlap=0.5,           # 50% window overlap doubles training samples
        random_offset=True,    # per-file jitter so each epoch sees different slices
    )

    # Parallel workers decode mp3 -> raw waveform on CPU. DAC tokenization
    # happens once per batch in the main process on GPU. Workers never touch
    # CUDA (no DAC, no model), so there's only one CUDA context in the system.
    # persistent_workers keeps the 4 worker processes alive across iterator
    # restarts, paying the ~1s startup cost only once.
    dataloader = DataLoader(
        raw_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=RawAudioDataset.collate_fn,
    )

    # Validation dataloader: single-process is fine — runs in short bursts of
    # ~10 batches per trigger, parallelism not worth the worker overhead.
    val_raw_dataset = RawAudioDataset(
        audio_dir=config["validation_dir"],
        num_chunks=num_chunks,
        shuffle=False,
    )
    val_dataloader = DataLoader(
        val_raw_dataset,
        batch_size=config["batch_size"],
        num_workers=0,
        pin_memory=True,
        collate_fn=RawAudioDataset.collate_fn,
    )


    # scaler = torch.amp.GradScaler()  # before training loop (disabled)

    prev_iter_end = None

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        if global_step >= config["training_steps"]:
            break

        for batch_idx, batch_waveforms in enumerate(dataloader):
            iter_start_time = time.time()
            # Wall-clock gap since previous iteration ended = how long workers
            # took to produce the next raw waveform batch (mp3 decode etc).
            dataload_time = (
                iter_start_time - prev_iter_end if prev_iter_end is not None else 0.0
            )

            # Dataloader gives us raw waveforms [B, 1, samples] from CPU
            # workers. Tokenize on GPU here, in the main process. This is the
            # only DAC call per step.
            batch_waveforms = batch_waveforms.to(device, non_blocking=True)
            with torch.no_grad():
                codes = tokenizer.encode(batch_waveforms)  # [B, K, T]

            past_tokens = codes[:, :, : config["past_len"]]
            future_tokens = codes[:, :, config["past_len"]:]

            batch_size_curr, n_cb, total_time = past_tokens.shape

            if future_tokens.shape[-1] < config["future_len"]:
                print(
                    f"Warning: Batch {batch_idx} has insufficient tokens, skipping - "
                    f"{future_tokens.shape[-1]}<{config['future_len']}"
                )
                continue

            future_tokens = future_tokens[:, :, : config["future_len"]].contiguous()

            # Model expects [B, T, K]; tokenizer returns [B, K, T].
            past_tokens = past_tokens.transpose(1, 2).long()
            future_tokens = future_tokens.transpose(1, 2).long()

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

            is_metric_step = global_step % config["log_metrics_every"] == 0

            # bf16 autocast: same exponent range as fp32, so no GradScaler needed.
            # fp16 without GradScaler silently kills training via gradient underflow.
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                if is_metric_step:
                    loss, train_metrics = model.get_training_loss(
                        past_tokens=past_tokens,
                        future_tokens=future_tokens,
                        return_metrics=True,
                    )
                else:
                    loss = model.get_training_loss(
                        past_tokens=past_tokens,
                        future_tokens=future_tokens,
                    )
                    train_metrics = None

            loss_val = loss.item()
            # Use standard backward()/optimizer.step() instead of GradScaler
            loss.backward()

            # Gradient clipping
            if config["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["gradient_clip"]
                )

            # Standard optimizer step (no GradScaler)
            optimizer.step()
            scheduler.step()  # update lr scheduler

            # Pure train-step time = forward + backward + clip + optimizer + scheduler.
            # Logging blocks below are NOT counted here so we can spot pure model
            # drift independently of audio/validation overhead spikes.
            train_step_time = time.time() - iter_start_time

            # =====================================================================
            # 7. Logging
            # =====================================================================
            epoch_loss += loss_val
            valid_batches += 1

            # Log metrics every N batches
            if is_metric_step:
                grad_norm = math.sqrt(
                    sum(
                        p.grad.norm().item() ** 2
                        for p in model.parameters()
                        if p.grad is not None
                    )
                )

                log_metrics(
                    writer=writer,
                    loss=loss_val,
                    lr=scheduler.get_last_lr()[0],
                    global_step=global_step,
                    grad_norm=grad_norm,
                )
                if train_metrics is not None:
                    log_codebook_metrics(
                        writer, train_metrics, global_step, prefix="Train"
                    )

                top1_mean = (
                    train_metrics["Top1"].mean().item()
                    if train_metrics is not None
                    else float("nan")
                )
                print(
                    f"Epoch {epoch + 1} | step: {global_step:>5d} | "
                    f"Loss: {loss_val:.4f} | Top1: {top1_mean:.3f} | "
                    f"Grad: {grad_norm:.3f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            # Exposure-bias check: AR-with-teacher-forcing vs free-running AR.
            # Expensive (two full AR generations); fires every log_exposure_every steps.
            if global_step % config["log_exposure_every"] == 0:
                model.eval()
                with torch.no_grad():
                    preds_tf = model.predict(
                        prompt_tokens=past_tokens,
                        true_future_tokens=future_tokens,
                        predict_by_one=True,
                    )
                    preds_ar = model.predict(
                        prompt_tokens=past_tokens,
                        temperature=1.0,
                        top_k=200,
                        top_p=0.95,
                        repetition_penalty=1.1,
                        predict_by_one=False,
                    )

                    acc_tf_per_cb = (preds_tf == future_tokens).float().mean(dim=(0, 1))
                    acc_ar_per_cb = (preds_ar == future_tokens).float().mean(dim=(0, 1))
                    log_codebook_metrics(
                        writer,
                        {
                            "AccTF": acc_tf_per_cb,
                            "AccAR": acc_ar_per_cb,
                            "ExposureGap": acc_tf_per_cb - acc_ar_per_cb,
                        },
                        global_step,
                        prefix="Generation",
                    )

                    del preds_tf, preds_ar
                model.train()

            # Validation pass — use distinct variable names so we don't clobber
            # the training tensors used by the audio block below.
            if global_step % config["validation_every"] == 0:
                model.eval()
                val_loss_sum = 0.0
                val_per_cb_loss = torch.zeros(config["n_codebooks"], device=device)
                val_per_cb_top1 = torch.zeros(config["n_codebooks"], device=device)
                val_batches_seen = 0
                with torch.no_grad():
                    for val_waveforms in val_dataloader:
                        val_waveforms = val_waveforms.to(device, non_blocking=True)
                        val_codes = tokenizer.encode(val_waveforms)  # [B, K, T]
                        val_past = val_codes[:, :, : config["past_len"]]
                        val_future = val_codes[:, :, config["past_len"]:]

                        if val_future.shape[-1] < config["future_len"]:
                            continue

                        val_future = val_future[:, :, : config["future_len"]].contiguous()
                        val_past = val_past.transpose(1, 2).long()
                        val_future = val_future.transpose(1, 2).long()

                        vloss, vmetrics = model.get_training_loss(
                            past_tokens=val_past,
                            future_tokens=val_future,
                            return_metrics=True,
                        )
                        val_loss_sum += vloss.item()
                        val_per_cb_loss += vmetrics["Loss"].float()
                        val_per_cb_top1 += vmetrics["Top1"].float()
                        val_batches_seen += 1
                        if val_batches_seen >= 10:
                            break

                if val_batches_seen > 0:
                    writer.add_scalar(
                        "Val/Loss", val_loss_sum / val_batches_seen, global_step
                    )
                    log_codebook_metrics(
                        writer,
                        {
                            "Loss": val_per_cb_loss / val_batches_seen,
                            "Top1": val_per_cb_top1 / val_batches_seen,
                        },
                        global_step,
                        prefix="Val",
                    )

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

                    log_generation_stats(
                        writer=writer,
                        predictions=predictions_autoreg,
                        global_step=global_step,
                        prefix="Generation",
                    )

                    del predictions_autoreg, gt_waveform, pred_waveform
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    model.train()

            iter_end_time = time.time()
            iter_time = iter_end_time - iter_start_time          # full body, incl. logging
            logging_time = iter_time - train_step_time            # AR gen + val + audio
            prev_iter_end = iter_end_time

            if is_metric_step:
                writer.add_scalar("Train/IterTime", iter_time, global_step)
                writer.add_scalar("Train/TrainStepTime", train_step_time, global_step)
                writer.add_scalar("Train/DataLoadTime", dataload_time, global_step)
                writer.add_scalar("Train/LoggingTime", logging_time, global_step)
                writer.add_scalar(
                    "Train/TimePerSample",
                    train_step_time / config["batch_size"],
                    global_step,
                )
                print(
                    f"  timings (s): iter={iter_time:.2f}  "
                    f"train={train_step_time:.2f}  data={dataload_time:.2f}  "
                    f"log={logging_time:.2f}"
                )

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
