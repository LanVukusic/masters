import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def generate_dj_waveform(audio_tensor, width=500):
    """Generate DJ-style waveform (peak-holding amplitude envelope)."""
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()
    waveform = (
        audio_tensor.mean(dim=0) if audio_tensor.shape[0] > 1 else audio_tensor[0]
    )
    waveform = torch.abs(waveform).unsqueeze(0).unsqueeze(0)
    kernel_size = max(1, waveform.shape[-1] // width)
    return F.max_pool1d(waveform, kernel_size).squeeze().detach().cpu()


def generate_spectrogram(audio_tensor, sample_rate=24000, n_fft=2048, n_mels=128):
    """Generate mel-spectrogram for audio visualization."""
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()
    waveform = (
        audio_tensor.mean(dim=0) if audio_tensor.shape[0] > 1 else audio_tensor[0]
    )

    transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        n_mels=n_mels,
        power=2.0,
    )
    spec = transform(waveform.unsqueeze(0)).squeeze(0)
    spec = torch.log10(spec.clamp(min=1e-10) * 10) * 10
    return spec.detach().cpu()


def _plot_waveform(ax, data, label, color):
    ax.plot(data.detach().cpu().numpy(), label=label, color=color, linewidth=0.5)


def create_waveform_comparison(gt_waveform, pred_waveform, step, width=500):
    """Create waveform comparison figure for TensorBoard."""
    fig = Figure(figsize=(12, 4), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    _plot_waveform(
        ax, generate_dj_waveform(gt_waveform, width), "Ground Truth", "green"
    )
    _plot_waveform(ax, generate_dj_waveform(pred_waveform, width), "Prediction", "red")
    ax.set_title(f"Waveform (Step {step})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_spectrogram_comparison(gt_waveform, pred_waveform, step, sample_rate=24000):
    """Create spectrogram comparison figure for TensorBoard."""
    import numpy as np

    spec_gt = generate_spectrogram(gt_waveform, sample_rate)
    spec_pred = generate_spectrogram(pred_waveform, sample_rate)

    fig = Figure(figsize=(12, 8), dpi=100)
    canvas = FigureCanvas(fig)

    for i, (spec, title) in enumerate(
        [
            (spec_gt, f"Ground Truth (Step {step})"),
            (spec_pred, "Prediction"),
        ],
        1,
    ):
        ax = fig.add_subplot(311 if i == 1 else 312)
        ax.imshow(
            spec.detach().cpu().numpy(), origin="lower", aspect="auto", cmap="viridis"
        )
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

    ax = fig.add_subplot(313)
    delta = (spec_gt - spec_pred).detach().cpu().numpy()
    vmax = np.abs(delta).max()
    ax.imshow(delta, origin="lower", aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_title("Delta (GT - Pred)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    return fig
