import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def generate_dj_waveform(audio_tensor, width=1000):
    """Generate DJ-style waveform (peak-holding amplitude envelope).

    Args:
        audio_tensor: (channels, samples) - can be GPU or CPU tensor
        width: Desired output length (pixels/bars)
    Returns:
        Downsampled amplitude envelope (width,) - CPU tensor
    """
    # Ensure we're working with CPU tensor to avoid GPU memory retention
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()

    waveform = (
        audio_tensor.mean(dim=0) if audio_tensor.shape[0] > 1 else audio_tensor[0]
    )
    waveform = torch.abs(waveform).unsqueeze(0).unsqueeze(0)
    kernel_size = max(1, waveform.shape[-1] // width)
    return F.max_pool1d(waveform, kernel_size).squeeze().detach().cpu()


def generate_spectrogram(
    audio_tensor,
    sample_rate=16000,
    n_fft=400,
    hop_length=None,
    n_mels=None,
    power=2.0,
    db_scale=True,
):
    """Generate spectrogram or mel-spectrogram.

    Args:
        audio_tensor: (channels, samples) - can be GPU or CPU tensor
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: If set, returns Mel-spectrogram instead of linear
        power: Exponent for spectrogram (1=amplitude, 2=power)
        db_scale: Convert to decibels
    Returns:
        Spectrogram tensor (freq, time) or (mel, time) - CPU tensor
    """
    # Ensure we're working with CPU tensor to avoid GPU memory retention
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()

    waveform = (
        audio_tensor.mean(dim=0) if audio_tensor.shape[0] > 1 else audio_tensor[0]
    )

    if n_mels:
        transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            center=True,
            pad_mode="reflect",
        )
    else:
        transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            center=True,
            pad_mode="reflect",
        )

    spec = transform(waveform.unsqueeze(0)).squeeze(0)

    if db_scale:
        spec = torch.log10(spec.clamp(min=1e-10) * 10) * 10  # Simple dB conversion
    return spec.detach().cpu()


def _plot_spectrogram(ax, spec, title, cmap="viridis"):
    """Helper: plot spectrogram on given axis."""
    import numpy as np

    spec_np = spec.detach().cpu().numpy()
    im = ax.imshow(spec_np, origin="lower", aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    return im


def _plot_waveform(ax, data, label, color, alpha=0.8):
    """Helper: plot waveform on given axis."""
    ax.plot(
        data.detach().cpu().numpy(),
        label=label,
        color=color,
        alpha=alpha,
        linewidth=0.5,
    )


def create_waveform_comparison(gt_waveform, pred_waveform, step, width=500):
    """Create DJ waveform comparison figure for TensorBoard."""
    fig = Figure(figsize=(12, 4), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    dj_gt = generate_dj_waveform(gt_waveform, width=width)
    dj_pred = generate_dj_waveform(pred_waveform, width=width)

    _plot_waveform(ax, dj_gt, "Ground Truth", "green")
    _plot_waveform(ax, dj_pred, "Prediction", "red")

    ax.set_title(f"DJ Waveform Comparison (Step {step})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_spectrogram_comparison(
    gt_waveform, pred_waveform, step, sample_rate=16000, n_mels=96, width=500
):
    """Create spectrogram comparison figure for TensorBoard."""
    fig = Figure(figsize=(12, 8), dpi=100)
    canvas = FigureCanvas(fig)

    # Generate spectrograms
    spec_gt = generate_spectrogram(gt_waveform, sample_rate, n_mels=n_mels)
    spec_pred = generate_spectrogram(pred_waveform, sample_rate, n_mels=n_mels)

    # Plot GT
    ax1 = fig.add_subplot(311)
    _plot_spectrogram(ax1, spec_gt, f"Ground Truth (Step {step})")

    # Plot Pred
    ax2 = fig.add_subplot(312, sharex=ax1)
    _plot_spectrogram(ax2, spec_pred, "Prediction")

    # Plot Delta
    ax3 = fig.add_subplot(313, sharex=ax1)
    delta = spec_gt - spec_pred
    import numpy as np

    delta_np = delta.detach().cpu().numpy()
    vmax = np.abs(delta_np).max()
    ax3.imshow(
        delta_np, origin="lower", aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax
    )
    ax3.set_title("Delta (GT - Pred)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Frequency")

    fig.tight_layout()
    return fig


def create_combined_visualization(
    gt_waveform, pred_waveform, step, sample_rate=16000, waveform_width=500, n_mels=96
):
    """Create combined waveform + spectrogram comparison for TensorBoard."""
    fig = Figure(figsize=(14, 6), dpi=100)
    canvas = FigureCanvas(fig)

    # Waveforms (top)
    ax1 = fig.add_subplot(211)
    dj_gt = generate_dj_waveform(gt_waveform, width=waveform_width)
    dj_pred = generate_dj_waveform(pred_waveform, width=waveform_width)
    _plot_waveform(ax1, dj_gt, "GT", "green")
    _plot_waveform(ax1, dj_pred, "Pred", "red")
    ax1.set_title(f"Waveform Comparison (Step {step})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Spectrograms (bottom)
    ax2 = fig.add_subplot(212)
    spec_gt = generate_spectrogram(gt_waveform, sample_rate, n_mels=n_mels)
    spec_pred = generate_spectrogram(pred_waveform, sample_rate, n_mels=n_mels)
    _plot_spectrogram(ax2, spec_gt, "GT Spectrogram")

    fig.tight_layout()
    return fig


# TensorBoard usage example:
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter("runs/audio_exp")
#
# fig = create_combined_visualization(gt, pred, step=global_step)
# writer.add_figure("audio/visualization", fig, global_step=global_step)
# writer.close()  # Important: close figure to free memory
