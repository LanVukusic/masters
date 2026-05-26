import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

from model_training.model_config import TARGET_SAMPLING_RATE

# Simple spectrogram generation (with padding for short audio)
def generate_spectrogram(audio_tensor, sample_rate=TARGET_SAMPLING_RATE, n_fft=2048, n_mels=128):
    waveform = audio_tensor.cpu().flatten()
    if waveform.shape[-1] < n_fft:
        waveform = F.pad(waveform, (0, n_fft - waveform.shape[-1]))
    
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=n_fft//4, n_mels=n_mels, power=2.0)
    spec = transform(waveform.unsqueeze(0)).squeeze(0)
    spec = torch.log10(spec.clamp(min=1e-10) * 10) * 10
    return spec.detach().cpu()

# Simple spectrogram comparison plot
def plot_spectrogram_comparison(gt_waveform, pred_waveform, step):
    spec_gt = generate_spectrogram(gt_waveform)
    spec_pred = generate_spectrogram(pred_waveform)
    
    fig = Figure(figsize=(10, 6), dpi=100)
    canvas = FigureCanvas(fig)
    
    for i, (spec, title) in enumerate([(spec_gt, f"Ground Truth (Step {step})"), (spec_pred, "Prediction")], 1):
        ax = fig.add_subplot(1, 2, i)
        ax.imshow(spec.numpy(), aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Frequency")
    
    fig.tight_layout()
    return fig

# Simple audio logging
def log_audio(writer, tokenizer, future_tokens, predictions_autoreg, global_step, future_len, audio_log_level):
    codes = future_tokens[:, :future_len, :audio_log_level].transpose(1, 2)
    gt_waveform = tokenizer.decode(codes)[0].cpu().flatten()
    writer.add_audio("Audio/GroundTruth", gt_waveform, global_step, tokenizer.sample_rate)
    
    codes_pred = predictions_autoreg[:, :future_len, :audio_log_level].transpose(1, 2)
    pred_waveform = tokenizer.decode(codes_pred)[0].cpu().flatten()
    writer.add_audio("Audio/Prediction", pred_waveform, global_step, tokenizer.sample_rate)
    
    return gt_waveform, pred_waveform

# Simple metrics logging
def log_metrics(writer, loss, lr, global_step, grad_norm=None):
    writer.add_scalar("Train/Loss", loss, global_step)
    writer.add_scalar("Train/LR", lr, global_step)
    if grad_norm:
        writer.add_scalar("Train/GradNorm", grad_norm, global_step)

# Simple visualization logging
def log_visualization(writer, gt_waveform, pred_waveform, global_step):
    fig = plot_spectrogram_comparison(gt_waveform, pred_waveform, global_step)
    writer.add_figure("Visualization/SpectrogramComparison", fig, global_step)
    import matplotlib.pyplot as plt
    plt.close(fig)


def log_codebook_metrics(writer, metrics, global_step, prefix="Train"):
    """
    Log per-codebook scalars + a mean across codebooks.
      metrics: dict[str, Tensor[K]] — e.g. {"Loss": ..., "Top1": ..., "Top5": ...}
      prefix:  TB tag prefix (e.g. "Train", "Val", "Generation")
    Produces tags of the form  {prefix}_per_cb/{short}/cb_{kk}  and  .../mean.
    """
    for short, values in metrics.items():
        values = values.detach().float()
        for k in range(values.shape[0]):
            writer.add_scalar(
                f"{prefix}_per_cb/{short}/cb_{k:02d}", values[k].item(), global_step
            )
        writer.add_scalar(
            f"{prefix}_per_cb/{short}/mean", values.mean().item(), global_step
        )


def log_generation_stats(writer, predictions, global_step, prefix="Generation"):
    """
    Log AR-generation health from a [B, T, K] long tensor of predicted tokens:
      - per-codebook token-id histograms ({prefix}/tokens_cb_KK)
      - per-codebook unique-token fraction and empirical entropy (per_cb scalars)

    Stuck-ness symptoms: unique fraction near 0, entropy near 0.
    """
    if predictions.dim() != 3:
        return
    preds = predictions.detach().cpu().long()
    _, _, K = preds.shape

    unique_fracs = torch.zeros(K)
    entropies = torch.zeros(K)
    for k in range(K):
        toks = preds[:, :, k].flatten()
        writer.add_histogram(f"{prefix}/tokens_cb_{k:02d}", toks, global_step, bins=64)

        n_total = toks.numel()
        unique_fracs[k] = toks.unique().numel() / max(n_total, 1)

        counts = torch.bincount(toks).float()
        probs = counts / counts.sum().clamp(min=1.0)
        entropies[k] = -(probs * probs.clamp(min=1e-12).log()).sum()

    log_codebook_metrics(
        writer,
        {"UniqueFrac": unique_fracs, "Entropy": entropies},
        global_step,
        prefix=prefix,
    )