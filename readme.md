# Project: Real-Time Audio Continuation for Downstream Analysis

## 1. Objective

Train a Transformer model to ingest past audio tokens and predict a **N-second future horizon** of audio tokens.
**Primary Goal:** Enable downstream tasks (beat detection, coarse rendering) in real-time.
**Secondary Goal:** High-fidelity generation is NOT the priority; structural consistency and low latency are.

## 2. Core Design Decisions (Critical)

- **Architecture:** **Non-Autoregressive (NAR)**.
  - *Reasoning:* AR is too slow for 10s horizon (500+ sequential steps) and prone to drift. NAR allows single-pass forecasting.
- **Fidelity Strategy:** **Hierarchical RVQ Depth**.
  - **Codebook 1 (Coarse/Rhythm):** Predict full 10s horizon.
  - **Codebooks 2-N (Fine/Timbre):** Predict only immediate future (e.g., 1-2s) or skip for distant future.
  - *Reasoning:* Distant future is uncertain; prioritize rhythm over timbre for long horizons.
- **Latency Constraint:** Inference must be **real-time** (single forward pass < chunk duration).

## 3. Model Architecture

- **Backbone:** Decoder-only Transformer.
- **Input:** Context window of past audio tokens (e.g., 10s).
- **Output:** Future tokens (e.g., 10s).
- **Attention:** Causal mask on input; Full attention on output (NAR).
- **Heads:** Multiple LM heads, one per RVQ codebook layer.

## 4. Data & Codec

- **Codec:** **DAC (Discrete Audio Codec)**.
- **Tokenization:** RVQ encoded vectors.
- **Token Rate:** ~50Hz - 86Hz (approx. 500 tokens for 10s).
- **Preprocessing:** Audio chunks -> DAC Encoder -> RVQ Tokens.

## 5. Training & Loss

- **Loss Function:** Weighted Cross-Entropy.
- **Weighting Strategy:**
  - **Time Decay:** Higher weight for near future, lower for distant future.
  - **Depth Decay:** Higher weight for Codebook 1, lower for deeper codebooks at distant timesteps.
- **Optimizer:** AdamW.
- **Scheduler:** Cosine Decay.

## 6. Evaluation Metrics

- **Primary:** **Downstream Task Performance**.
  - Train a beat detector on real audio.
  - Test beat detector on *generated* continuation.
  - Metric: F1 Score of beat detection.
- **Secondary:**
  - **Codebook Accuracy:** Per-layer accuracy over time horizon.
  - **Latency:** Wall-clock inference time (ms).
- **Not Priority:** FAD (Fréchet Audio Distance) or listening tests (unless needed for debugging).

## 8. AI Development Guidelines

When assisting with code generation or refactoring:

1. **DO NOT suggest Autoregressive loops** for the generation phase.
2. **Prioritize latency:** Avoid unnecessary operations in the forward pass.
3. **Respect RVQ Structure:** Ensure output shapes match `[Batch, Time, Codebook_Layers]`.
4. **Focus on NAR Stability:** If suggesting loss functions, include mechanisms to prevent mode collapse (e.g., classifier-free guidance or specific smoothing).
