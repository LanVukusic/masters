#!/usr/bin/env python3
"""
Diagnostic - test raw DAC API with torchcodec for loading.
"""

import os
import dac
import torch
from torchcodec.decoders import AudioDecoder
from torch.utils.tensorboard import SummaryWriter


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load DAC model directly
    print("\n=== Loading DAC Model ===")
    model_path = dac.utils.download("24khz")
    model = dac.DAC.load(model_path)
    model.to(device)
    model.eval()
    print(f"Loaded: {model_path}")

    # Load audio file with torchcodec
    audio_path = (
        "dataset_gen/free_music/rotormotor/mp3s/001 Guy Contact - Cool Blue Liquid.mp3"
    )
    target_sr = 24000
    target_duration = 3.0

    print(f"\n=== Loading Audio: {os.path.basename(audio_path)} ===")
    decoder = AudioDecoder(audio_path, sample_rate=target_sr)
    samples = decoder.get_samples_played_in_range(
        start_seconds=2, stop_seconds=2+target_duration
    )
    waveform = samples.data  # [channels, samples] at 24kHz

    print(f"Loaded: {waveform.shape[1] / target_sr:.2f}s at {target_sr}Hz")
    print(f"Shape: {waveform.shape}")

    # Convert stereo to mono (torchcodec: dim=0 is channels)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, samples]

    # Add batch dimension: [1, samples] -> [1, 1, samples] for DAC preprocess
    waveform = waveform.unsqueeze(0)  # [1, 1, samples]

    print(f"After mono: {waveform.shape}")

    # Encode
    print("\n=== Encoding ===")
    waveform = waveform.to(device)
    # x = model.preprocess(waveform, target_sr)
    # print(f"Preprocessed: {waveform.shape}")

    _, codes, latents, _, _ = model.encode(waveform, n_quantizers=8)
    print(f"Codes shape: {codes.shape}")  # [batch, n_quantizers, time]
    
    z_out = model.quantizer.from_codes(codes)
    z = z_out[0] if isinstance(z_out, tuple) else z_out
    print(f"Z shape: {z.shape}")  # [batch, n_quantizers, time]

    # Decode
    print("\n=== Decoding ===")
    with torch.no_grad():
        decoded = model.decode(z)
    print(f"Decoded: {decoded.shape}")

    # Log to TensorBoard
    print("\n=== Logging ===")
    writer = SummaryWriter(log_dir="runs/diagnostic_raw_dac")

    # Input audio: [1, 1, samples] -> [samples]
    input_audio = waveform[0, 0].cpu()
    writer.add_audio("Input", input_audio, 0, sample_rate=target_sr)
    print(f"✓ Logged input: {input_audio.shape}")

    # Output audio
    output_audio = decoded[0, 0].cpu()  # [batch, channels, samples] -> [samples]
    writer.add_audio("Output", output_audio, 0, sample_rate=target_sr)
    print(f"✓ Logged output: {output_audio.shape}")

    writer.close()
    print("\n=== Done ===")
    print("Run: tensorboard --logdir runs/diagnostic_raw_dac")


if __name__ == "__main__":
    main()
