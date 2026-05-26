# Training Metrics Reference

What every TensorBoard tag in `train.py` actually means, what its range is, and
what bad values look like.

All "good/bad" reference values assume `vocab_size = 1024` and the project's
12-codebook DAC tokens. Adjust if you change the vocab.

## Quick reference: random baselines

These are the values you would see from a model that learned nothing — useful
to know whether early-training numbers represent real progress.

| Quantity                 | Random baseline (per codebook) |
| ------------------------ | ------------------------------ |
| Cross-entropy loss       | `ln(1024) ≈ 6.93`              |
| Top-1 accuracy           | `1/1024 ≈ 0.001` (0.1%)        |
| Top-5 accuracy           | `5/1024 ≈ 0.005` (0.5%)        |
| Empirical entropy (nats) | up to `ln(1024) ≈ 6.93`        |
| Unique-token fraction    | up to 1.0 (capped by samples)  |

A codebook whose `Loss` plateau is `~6.9` and whose `Top1` plateau is `~0.001`
is effectively producing uniform-random predictions. It contributes nothing
useful to the decoded audio.

---

## Training-step scalars (every `log_metrics_every` steps)

### `Train/Loss`

Mean over the 12 codebooks of the per-codebook cross-entropy. This is also the
quantity being minimized.

- Range: `[0, ~6.93]`. Lower is better.
- Sanity early on: should fall from near `6.93` toward `4–5` within a few
  hundred steps if anything is working.
- Plateauing at `6.93` exactly means the model is not learning. Plateauing well
  below that but with `Top1` still near random means it is producing diffuse
  distributions — see `Top1` below.

### `Train/LR`

Current learning rate from the scheduler. Linear warmup over
`num_warmup_steps`, then linear decay to 0 over `training_steps`. Logged so you
can correlate loss spikes with the warmup ramp, or notice if the schedule was
configured wrong.

### `Train/GradNorm`

Global `L2` norm of all parameter gradients, computed after clipping:

```
sqrt( sum_p ||p.grad||_2^2 )
```

- Healthy range: roughly the order of magnitude of `gradient_clip` (currently
  `30.0`). For this model, a settled value of `1–10` is typical; the early
  warmup phase usually shows higher norms.
- Persistent spikes near the clip ceiling: the clip is the only thing keeping
  training stable; consider lowering LR.
- Drifting to near 0 with loss still high: vanishing gradients somewhere; the
  model is no longer learning.
- NaN / Inf: AMP underflow or a numerical bug. Training is broken — stop and
  investigate.

---

## Per-codebook training scalars (every `log_metrics_every` steps)

These live under `Train_per_cb/` and each one has 12 codebook entries
(`cb_00`–`cb_11`) plus a `mean` aggregator.

### `Train_per_cb/Loss/cb_NN`

Cross-entropy loss restricted to codebook `NN`:

```
CE_k = - mean_{b,t} log P_model(future[b,t,k] | context)
```

The averaged `Train/Loss` hides which codebooks are actually learning. This
breaks it out.

- Codebook 0 (coarsest) should drop fastest — it captures pitch / fundamental
  structure. Expect it under `4` within the first few hundred steps.
- Codebooks 8–11 (finest) are intrinsically harder — they model residual
  detail. Expect them to drop more slowly, and to plateau higher than
  codebook 0.
- A codebook stuck near `6.93` for the whole run is dead: its head is
  effectively untrained. Decoded audio gets random fine-detail content from
  it, which usually surfaces as wideband noise.

### `Train_per_cb/Top1/cb_NN`

Fraction of positions where the model's most likely predicted token equals the
ground-truth token, for that codebook:

```
Top1_k = mean_{b,t} [ argmax_v logits[b,t,k,v] == future[b,t,k] ]
```

- Random baseline: `0.001`.
- "Learning" threshold: `> 0.01` (10× random).
- "Healthy mid-training" for codebook 0: `0.1–0.3`. For codebook 11: anything
  meaningfully above `0.005` is real signal.
- `Top1/mean` is the most readable single number in the dashboard. Watch it
  trend over a run.

### `Train_per_cb/Top5/cb_NN`

Fraction of positions where the ground-truth token is among the 5 highest-logit
predictions. Same intuition as `Top1` but smoother and more forgiving — useful
when `Top1` is still very noisy.

- Random baseline: `0.005`.
- Top-5 usually leads top-1 by a wide margin. If `Top5` is close to `Top1`,
  the model is hyper-confident on a small set — possibly the precursor to a
  mode collapse.

---

## Validation scalars (every `validation_every` steps)

Same metrics as training, computed on up to 10 batches from `val_dataloader`,
with the model in `eval()` mode (dropout off).

### `Val/Loss`, `Val_per_cb/Loss/*`, `Val_per_cb/Top1/*`

Definitions match the training versions. The thing to watch is the **gap**
between `Train/Loss` and `Val/Loss`:

- `Val/Loss` falling alongside `Train/Loss`: model is generalizing.
- `Val/Loss` flat or rising while `Train/Loss` falls: classic overfit.
  Likely on small datasets; rotormotor/mp3s_small is exactly the regime this
  happens in.
- `Val/Loss` and `Train/Loss` both flat: model isn't learning, problem is on
  the optimization side, not the data side.

---

## Exposure-bias scalars (every `log_exposure_every` steps)

These come from running the AR `predict` method in two modes on the same batch
and comparing per-codebook agreement with the ground-truth future.

### `Generation_per_cb/AccTF/cb_NN`

Accuracy of `predict(..., predict_by_one=True)`. At each step the model
predicts the next frame, then the ground-truth frame is appended to the
context. So at every step the conditioning is correct; this measures next-step
quality given perfect history.

- With the CLM shift fixed, `AccTF` should track `Train_per_cb/Top1` closely
  — same task, just measured at inference time. Disagreement here means the
  training and inference code paths diverge somehow.

### `Generation_per_cb/AccAR/cb_NN`

Accuracy of `predict(..., predict_by_one=False)`. Standard autoregressive
sampling — the model's own outputs feed back as context. Errors accumulate,
so this is always lower than `AccTF`.

- `AccAR` will start very low (model has no idea what's coming next; sampling
  is sound but uncorrelated with this specific GT future) and may not exceed
  `0.01` even for a well-trained model. That is **normal** — there is no
  unique ground-truth continuation of a music clip. Track its trend, not its
  absolute value.

### `Generation_per_cb/ExposureGap/cb_NN`

`AccTF - AccAR` per codebook. This is the **train/inference mismatch
indicator**.

- Some gap is expected — `AccTF` is much easier than `AccAR`. A widening
  gap over training is the diagnostic: it means the model's next-token
  prediction is getting better but its joint-distribution generation is
  not. That was the exact failure mode of the buggy conformer.
- Stable or shrinking gap: the model's improvements are transferring to AR
  generation as well. This is what you want.

---

## Generation-health scalars (every `log_audio_every` steps)

Computed from the AR-generated token tensor `predictions_autoreg`
(`[B, T_future, K]`). They describe the *output distribution* the model
actually produces during free-running generation.

### `Generation_per_cb/UniqueFrac/cb_NN`

```
UniqueFrac_k = (# distinct token ids predicted for codebook k) / (B * T_future)
```

Bounded above by `min(1.0, vocab_size / (B * T_future))`. For a batch of 2 and
`T_future = 150`, the cap is `1.0` (vocab=1024 >> 300 samples).

- A healthy generation should use *many* distinct tokens — DAC tokens encode
  varying audio, so coarse codebooks naturally spread across dozens to
  hundreds of token ids per second of audio.
- `UniqueFrac/cb_00 → 0` means the model is emitting the same coarse token
  repeatedly: single sustained pitch, "buzz", or silence in the decoded audio.
- `UniqueFrac/cb_11 → 0` is less audible but still a sign of fine-detail
  collapse.

### `Generation_per_cb/Entropy/cb_NN`

Empirical entropy (in nats) of the realized predictions for codebook `k`:

```
H_k = - sum_v p_k(v) log p_k(v),   p_k(v) = count_k(v) / N
```

This is the entropy of the **empirical** distribution over generated tokens,
not the model's per-step output distribution. It captures whether the model
explores or repeats.

- Range: `[0, ln(1024) ≈ 6.93]`.
- A natural music distribution over DAC tokens has entropy in roughly
  `4–6` nats per codebook (rough order of magnitude, varies with content).
- `Entropy → 0` is the unambiguous mode-collapse signal.
- Entropy dropping while `Train/Top1` keeps rising is the precursor: the
  model is becoming too confident on too few tokens.

---

## Histograms (every `log_audio_every` steps)

### `Generation/tokens_cb_NN`

Distribution of generated token ids for each codebook, plotted as a TB
histogram. Twelve histograms per logging step; you can collapse the folder in
the TB UI if it's noisy.

What to look for over time:

- Wide, well-spread distribution: model is producing varied tokens.
- Narrowing to a single spike: matches a `UniqueFrac → 0` collapse; the spike
  identifies *which* token id the model fell into.
- A few sharp modes: model has memorized a small vocabulary subset; common
  for overfit runs on tiny datasets.
- Compare across codebooks: codebook 0 should look "music-like" (a few wider
  peaks for common pitches); higher codebooks should look closer to uniform
  random because they encode residual noise.

---

## Audio and figures (every `log_audio_every` steps)

### `Audio/GroundTruth`

The actual future portion of the batch, decoded by the DAC tokenizer. Lives in
TB's Audio panel; click play. This is your reference for what the prediction
*should* sound like.

### `Audio/Prediction`

The AR-generated continuation from the same prompt, decoded by the same
tokenizer. Use this in conjunction with the spectrogram to judge subjective
quality. Numerical metrics can lie about audio quality in either direction —
listening is the final test.

### `Visualization/SpectrogramComparison`

Side-by-side log-mel spectrograms of `GroundTruth` and `Prediction`. Useful
because:

- Visible horizontal bands in the prediction = sustained tones / buzz.
- Visible vertical bands = clicks / impulse artifacts.
- "Smeared" / featureless spectrogram = the model is producing low-energy
  noise.
- The ground-truth spectrogram is also useful as a sanity check that the
  tokenizer round-trip itself isn't lossy — if GT looks bad too, the problem
  is upstream of the model.

---

## Failure-mode → metric cheatsheet

If audio output sounds like…             | First check these
---                                       | ---
**Single sustained tone or buzz**         | `Generation_per_cb/UniqueFrac/cb_00`, `Entropy/cb_00`, `Generation/tokens_cb_00` histogram
**Wideband hissy noise**                  | `Train_per_cb/Loss/cb_8..11` (are they flat at 6.93?), `Generation_per_cb/UniqueFrac/cb_8..11`
**Coherent but musically random**         | This may actually be fine — there is no unique correct continuation. Compare `AccTF` vs `AccAR`.
**Output identical to prompt's tail**     | `Generation/tokens_cb_NN` histograms vs the prompt's tokens; mode collapse to the most recent context
**Trained well, then suddenly degraded**  | `Train/GradNorm` (look for a recent spike), `Train/Loss` (look for a step-change), `Train/LR` (verify scheduler)
**Trains fine, generates garbage**        | `Generation_per_cb/ExposureGap/mean` — this is the train/inference mismatch indicator
**Train loss falls, val loss rises**      | Overfit. Reduce model size, add data, or stop earlier.
