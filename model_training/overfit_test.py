"""
Overfit-a-single-batch diagnostic.

Definitively answers "can my model learn anything?". We grab one batch, then
feed it to the model 500 times. A healthy model + training loop will drive the
loss to near zero — the model literally memorizes the samples. If loss
plateaus above ~4.0, something is structurally broken (model, loss, optimizer,
or shape contract).

Run:
    uv run python model_training/overfit_test.py

Healthy trajectory (rough, for K=4 codebooks):
    step    0  loss ~ 6.93   (random)
    step  100  loss <  2.0
    step  300  loss <  0.5
    step  500  loss <  0.1

If loss stops dropping above 4.0, the bug is in the model code, not the data
or training budget.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from models.simple import AudioContinuationTransformer
from model_training.dataloader.IterableDataset import (
    RawAudioDataset,
    TokenizedAudioDataset,
)
from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer
from model_training.model_config import (
    MODEL_CONFIG,
    DAC_FRAME_SIZE,
    compute_token_lengths,
    tokens_to_chunks,
)


AUDIO_DIR = "dataset_gen/free_music/rotormotor/mp3s"
BATCH_SIZE = 4
NUM_STEPS = 500
LEARNING_RATE = 1e-3
LOG_EVERY = 20


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = {**MODEL_CONFIG, "batch_size": BATCH_SIZE}
    config["past_len"], config["future_len"] = compute_token_lengths(DAC_FRAME_SIZE)

    print("Building model...")
    model = AudioContinuationTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")
    print(
        f"  d_model={config['d_model']}, n_layers={config['n_layers']}, "
        f"d_ff={config['d_ff']}, n_codebooks={config['n_codebooks']}"
    )
    print(f"  past_len={config['past_len']}, future_len={config['future_len']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )

    print("Building tokenizer + dataset...")
    tokenizer = DACAudioTokenizer(num_quantizers=config["n_codebooks"], device=device)

    num_chunks = tokens_to_chunks(
        config["past_len"] + config["future_len"], DAC_FRAME_SIZE
    )
    raw_dataset = RawAudioDataset(
        audio_dir=AUDIO_DIR,
        num_chunks=num_chunks,
        shuffle=False,
    )
    dataset = TokenizedAudioDataset(
        base_dataset=raw_dataset,
        tokenizer=tokenizer,
        past_chunks=config["past_len"],
        device=device,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        collate_fn=TokenizedAudioDataset.collate_fn,
    )

    print("Sampling one batch...")
    batch = next(iter(loader))
    past = batch["past"].to(device).transpose(1, 2).long()
    future = (
        batch["future"].to(device)[:, :, : config["future_len"]].transpose(1, 2).long()
    )
    print(f"  past:   {tuple(past.shape)}  (B, T_past, K)")
    print(f"  future: {tuple(future.shape)}  (B, T_future, K)")

    # Sanity check: a forward pass with the model in eval mode should give
    # near-uniform logits initially -> loss near ln(vocab).
    model.eval()
    with torch.no_grad():
        baseline_loss = model.get_training_loss(past, future)[0].item()
    print(
        f"\nInitial loss (eval mode): {baseline_loss:.4f}  "
        f"(random baseline ≈ {torch.tensor(float(config['vocab_size'])).log().item():.4f})"
    )

    print(f"\nOverfit run: {NUM_STEPS} steps on the same batch, lr={LEARNING_RATE}")
    print("=" * 60)

    model.train()
    losses = []
    for step in range(NUM_STEPS):
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = model.get_training_loss(past, future)[0]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)
        if step % LOG_EVERY == 0 or step == NUM_STEPS - 1:
            print(
                f"step {step:4d}  loss={loss_val:.4f}  grad_norm={grad_norm.item():.3f}"
            )

    print("=" * 60)
    final = losses[-1]
    initial = losses[0]
    print(f"\nInitial loss: {initial:.4f}")
    print(f"Final loss:   {final:.4f}")
    print(f"Reduction:    {initial - final:.4f}")

    if final < 0.5:
        verdict = "HEALTHY — model can learn. Architecture is fine."
    elif final < 2.0:
        verdict = "MOSTLY HEALTHY — model is learning but slowly. Check init / LR."
    elif final < 4.0:
        verdict = (
            "DEGRADED — model is partially stuck. Likely capacity or numerical issue."
        )
    else:
        verdict = "BROKEN — model cannot fit even one batch. Bug is structural."
    print(f"\nVerdict: {verdict}")
