#!/usr/bin/env python3
"""
Example usage of the enhanced AudioTokenizer with batch processing capabilities.
"""

import torch
import torchaudio
from model_training.model.audio_tokenizer import AudioTokenizer


def demonstrate_single_operations():
    """Demonstrate single audio file operations."""
    print("=== Single Operations Demo ===")

    # Initialize tokenizer
    tokenizer = AudioTokenizer(num_quantizers=8, device="cpu")

    # Create a dummy audio waveform for demonstration
    dummy_audio = torch.randn(1, 24000)  # 1 second at 24kHz

    # Single encoding from waveform
    encoded = tokenizer.encode_from_waveform(dummy_audio, 24000)
    print(f"Single encoded shape: {encoded.shape}")

    # Single decoding
    decoded = tokenizer.decode_to_waveform(encoded)
    print(f"Single decoded shape: {decoded.shape}")

    # Save single file
    tokenizer.decode_to_file(encoded, "single_demo.wav")
    print("Single file saved as 'single_demo.wav'")
    import os

    if os.path.exists("single_demo.wav"):
        os.remove("single_demo.wav")  # Clean up
    print()


def demonstrate_batch_operations():
    """Demonstrate batch operations."""
    print("=== Batch Operations Demo ===")

    tokenizer = AudioTokenizer(num_quantizers=8, device="cpu")

    # Create multiple dummy audio waveforms
    waveform1 = torch.randn(1, 24000)
    waveform2 = torch.randn(1, 24000)
    waveform3 = torch.randn(1, 24000)

    # Batch encoding from waveforms
    waveforms = [waveform1, waveform2, waveform3]
    encoded_batch = tokenizer.encode_from_waveform(waveforms, 24000)
    print(f"Batch encoded length: {len(encoded_batch)}")
    for i, encoded in enumerate(encoded_batch):
        print(f"  Batch item {i} shape: {encoded.shape}")

    # Batch decoding
    decoded_batch = tokenizer.decode_to_waveform(encoded_batch)
    print(f"Batch decoded length: {len(decoded_batch)}")
    for i, decoded in enumerate(decoded_batch):
        print(f"  Decoded item {i} shape: {decoded.shape}")

    # Batch file saving with multiple paths
    output_paths = ["batch_demo_0.wav", "batch_demo_1.wav", "batch_demo_2.wav"]
    tokenizer.decode_to_file(encoded_batch, output_paths)
    print(f"Batch files saved: {output_paths}")

    # Clean up
    import os

    for path in output_paths:
        if os.path.exists(path):
            os.remove(path)
    print()


def demonstrate_path_batch_operations():
    """Demonstrate batch operations with audio file paths."""
    print("=== Path Batch Operations Demo ===")

    tokenizer = AudioTokenizer(num_quantizers=4, device="cpu")

    # Create dummy audio files for demonstration
    dummy_audio1 = torch.randn(1, 12000)  # 0.5 seconds
    dummy_audio2 = torch.randn(1, 12000)  # 0.5 seconds

    # Save dummy files
    torchaudio.save("dummy1.wav", dummy_audio1, 24000)
    torchaudio.save("dummy2.wav", dummy_audio2, 24000)

    # Batch encoding from paths
    audio_paths = ["dummy1.wav", "dummy2.wav"]
    encoded_from_paths = tokenizer.encode_from_path(audio_paths)
    print(f"Encoded from paths length: {len(encoded_from_paths)}")
    for i, encoded in enumerate(encoded_from_paths):
        print(f"  Path {i} encoded shape: {encoded.shape}")

    # Clean up dummy files
    import os

    for path in ["dummy1.wav", "dummy2.wav"]:
        if os.path.exists(path):
            os.remove(path)
    print()


def demonstrate_mixed_operations():
    """Demonstrate mixed single and batch operations."""
    print("=== Mixed Operations Demo ===")

    tokenizer = AudioTokenizer(num_quantizers=4, device="cpu")

    # Single operation
    single_waveform = torch.randn(1, 12000)
    single_encoded = tokenizer.encode_from_waveform(single_waveform, 24000)
    print(f"Single encoded shape: {single_encoded.shape}")

    # Create batch from single (duplicate for demo)
    batch_encoded = [single_encoded, single_encoded, single_encoded]

    # Batch decoding
    batch_decoded = tokenizer.decode_to_waveform(batch_encoded)
    print(f"Mixed batch decoded length: {len(batch_decoded)}")

    # Batch file saving with numbered files
    tokenizer.decode_to_file(batch_encoded, "mixed_demo.wav")
    print(
        "Mixed batch saved with numbered files: mixed_demo_0.wav, mixed_demo_1.wav, mixed_demo_2.wav"
    )

    # Clean up
    import os

    for i in range(3):
        numbered_file = f"mixed_demo_{i}.wav"
        if os.path.exists(numbered_file):
            os.remove(numbered_file)
    print()


def main():
    """Run all demonstrations."""
    print("Enhanced AudioTokenizer - Batch Processing Demo\n")

    demonstrate_single_operations()
    demonstrate_batch_operations()
    demonstrate_path_batch_operations()
    demonstrate_mixed_operations()

    print("🎉 All demonstrations completed successfully!")
    print("\nKey Features:")
    print("- Single and batch encoding from paths or waveforms")
    print("- Single and batch decoding")
    print("- Batch file saving with multiple or numbered outputs")
    print("- CPU/GPU inference support")
    print("- Configurable number of quantizers")
    print("- Error handling for mismatched batch sizes")


if __name__ == "__main__":
    main()
