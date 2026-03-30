import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter
    from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer


def log_audio_samples(
    writer: "SummaryWriter",
    tokenizer: "DACAudioTokenizer",
    past_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    predictions_one_token: torch.Tensor,
    predictions_autoreg: torch.Tensor,
    global_step: int,
    future_len: int,
    audio_log_level: int = 4,
):
    """
    Log audio samples to TensorBoard including predictions and ground truth.

    Args:
        writer: TensorBoard SummaryWriter
        tokenizer: Audio tokenizer for decoding
        past_tokens: Past context tokens
        future_tokens: Ground truth future tokens
        predictions_one_token: Predictions using ground truth context
        predictions_autoreg: Full autoregressive predictions
        global_step: Current training step
        future_len: Length of future sequence
        audio_log_level: Number of codebooks to use for decoding

    Returns:
        Tuple of (gt_waveform, pred_waveform) for further visualization (CPU tensors)
    """
    try:
        # Log one-token prediction (using ground truth context)
        one_token_tokens = predictions_one_token[:, :future_len, :audio_log_level]
        waveform_one_token = tokenizer.decode_to_waveform(
            one_token_tokens.transpose(1, 2)
        )
        writer.add_audio(
            "Audio/OneTokenPrediction",
            waveform_one_token[0].cpu(),
            global_step,
            sample_rate=tokenizer.sampling_rate,
        )
        # Clear GPU memory immediately
        del waveform_one_token

        # Log full autoregressive prediction
        autoreg_tokens = predictions_autoreg[:, :future_len, :audio_log_level]
        waveform_autoreg = tokenizer.decode_to_waveform(autoreg_tokens.transpose(1, 2))
        writer.add_audio(
            "Audio/AutoregPrediction",
            waveform_autoreg[0].cpu(),
            global_step,
            sample_rate=tokenizer.sampling_rate,
        )

        # Log ground truth
        gt_tokens = future_tokens[:, :future_len, :audio_log_level]
        gt_waveform = tokenizer.decode_to_waveform(gt_tokens.transpose(1, 2))
        writer.add_audio(
            "Audio/GroundTruth",
            gt_waveform[0].cpu(),
            global_step,
            sample_rate=tokenizer.sampling_rate,
        )

        # Return CPU copies for visualization
        result = (gt_waveform[0].cpu(), waveform_autoreg[0].cpu())

        # Clean up GPU tensors
        del autoreg_tokens, gt_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"Warning: Could not decode audio for logging: {e}")
        return None, None


def log_training_metrics(
    writer: "SummaryWriter",
    loss: float,
    lr: float,
    global_step: int,
    grad_norm: float | None = None,
    teacher_forcing_ratio: float | None = None,
    logits: torch.Tensor | None = None,
    predicted_tokens: list | None = None,
    log_memory: bool = True,
):
    """
    Log training metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        loss: Current loss value
        lr: Current learning rate
        global_step: Current training step
        grad_norm: Current gradient norm (optional)
        teacher_forcing_ratio: Current teacher forcing ratio (optional)
        logits: Model output logits for entropy calculation (optional)
        predicted_tokens: List of predicted token IDs for unique token count (optional)
        log_memory: Whether to log CUDA memory usage
    """
    writer.add_scalar("Train/Loss", loss, global_step)
    writer.add_scalar("Train/LR", lr, global_step)

    if grad_norm is not None:
        writer.add_scalar("Train/GradNorm", grad_norm, global_step)

    if teacher_forcing_ratio is not None:
        writer.add_scalar(
            "Train/TeacherForcingRatio", teacher_forcing_ratio, global_step
        )

    if logits is not None:
        writer.add_scalar(
            "Metrics/Entropy", torch.special.entr(logits).mean().item(), global_step
        )

    if predicted_tokens is not None:
        writer.add_scalar(
            "Metrics/UniqueTokens", len(set(predicted_tokens)), global_step
        )

    if log_memory and torch.cuda.is_available():
        # Reset peak memory stats at start of step for accurate tracking
        torch.cuda.reset_peak_memory_stats()

        writer.add_scalar(
            "Memory/AllocatedMB",
            torch.cuda.memory_allocated() / 1024 / 1024,
            global_step,
        )
        writer.add_scalar(
            "Memory/ReservedMB",
            torch.cuda.memory_reserved() / 1024 / 1024,
            global_step,
        )
        writer.add_scalar(
            "Memory/MaxAllocatedMB",
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            global_step,
        )


def log_dj_waveform(
    writer: "SummaryWriter",
    gt_waveform: torch.Tensor,
    pred_waveform: torch.Tensor,
    global_step: int,
    sample_rate: int = 16000,
    width: int = 500,
):
    """
    Log DJ waveform comparison figure to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        gt_waveform: Ground truth waveform (CPU tensor)
        pred_waveform: Predicted waveform (CPU tensor)
        global_step: Current training step
        sample_rate: Audio sample rate
        width: Width for waveform generation
    """
    try:
        from utils.ploting import create_spectrogram_comparison

        fig = create_spectrogram_comparison(
            gt_waveform,
            pred_waveform,
            global_step,
            sample_rate=sample_rate,
            width=width,
        )
        writer.add_figure("Visualization/SpectrogramComparison", fig, global_step)
        import matplotlib.pyplot as plt

        plt.close(fig)
        del fig

        # Clear any remaining CUDA cache after matplotlib operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Could not generate spectrogram: {e}")
