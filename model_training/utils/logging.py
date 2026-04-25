import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter
    from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer


def log_audio_samples(
    writer: "SummaryWriter",
    tokenizer: "DACAudioTokenizer",
    future_tokens: torch.Tensor,
    predictions_one_token: torch.Tensor,
    predictions_autoreg: torch.Tensor,
    global_step: int,
    future_len: int,
    audio_log_level: int,
):
    """
    Log audio samples to TensorBoard.

    All tensor formats:
    - future_tokens: [B, T, K] (model format)
    - predictions: [B, T, K] (model format)
    - decode expects: [B, K, T]
    """
    try:
        sample_rate = tokenizer.sample_rate
        actual_codebooks = min(audio_log_level, future_tokens.shape[-1])

        def decode_and_prepare(tokens):
            """Decode tokens and prepare for TensorBoard [B, K, T] -> [B, 1, S] -> [S]"""
            codes = tokens[:, :future_len, :actual_codebooks].transpose(1, 2)
            
            #codes:    [1, num_quantizers, time_steps]
            waveform = tokenizer.decode(codes)
            if isinstance(waveform, list):
                waveform = waveform[0]
            if waveform.is_cuda:
                waveform = waveform.cpu()
            # [B, 1, S] -> take first sample -> [S]
            return waveform[0].flatten()

        # Decode and log each
        gt_waveform = decode_and_prepare(future_tokens)
        writer.add_audio("Audio/GroundTruth", gt_waveform, global_step, sample_rate)

        pred_waveform = decode_and_prepare(predictions_autoreg)
        writer.add_audio(
            "Audio/AutoregPrediction", pred_waveform, global_step, sample_rate
        )

        one_token_waveform = decode_and_prepare(predictions_one_token)
        writer.add_audio(
            "Audio/OneTokenPrediction", one_token_waveform, global_step, sample_rate
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return gt_waveform, pred_waveform

    except Exception as e:
        import traceback

        print(f"Warning: Could not decode audio for logging: {e}")
        traceback.print_exc()
        return None, None

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
    sample_rate: int = 24000,
    width: int = 500,
):
    """
    Log DJ waveform comparison figure to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        gt_waveform: Ground truth waveform (CPU tensor)
        pred_waveform: Predicted waveform (CPU tensor)
        global_step: Current training step
        sample_rate: Audio sample rate (default 24000 for DAC)
        width: Width for waveform generation
    """
    try:
        from utils.ploting import create_spectrogram_comparison

        fig = create_spectrogram_comparison(
            gt_waveform,
            pred_waveform,
            global_step,
            sample_rate=sample_rate,
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
