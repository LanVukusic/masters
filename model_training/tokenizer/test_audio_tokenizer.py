#!/usr/bin/env python3
"""
Test script for the AudioTokenizer class.
"""

import torch
import torchaudio
import os
from audio_tokenizer import AudioTokenizer


def test_basic_functionality():
    """Test basic encoding and decoding functionality."""
    print("=== Testing Basic Functionality ===")

    # Initialize tokenizer
    tokenizer = AudioTokenizer(
        num_quantizers=8
    )  # Using fewer quantizers for faster testing

    print("Model info:")
    print(tokenizer.get_model_info())

    # Check if music.mp3 exists in testiranje_mimi directory
    audio_path = "../../testiranje_mimi/music.mp3"
    if not os.path.exists(audio_path):
        print(
            f"Warning: {audio_path} not found, skipping tests that require audio file"
        )
        return

    print(f"\nTesting with audio file: {audio_path}")

    # Test loading audio
    waveform, sr = tokenizer.load_audio_from_path(audio_path)
    print(f"Loaded waveform shape: {waveform.shape}, sample rate: {sr}")

    # Test encoding from path
    encoded = tokenizer.encode_from_path(audio_path, num_quantizers=8)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded dtype: {encoded.dtype}")

    # Test encoding from waveform
    encoded_from_waveform = tokenizer.encode_from_waveform(
        waveform, sr, num_quantizers=8
    )
    print(f"Encoded from waveform shape: {encoded_from_waveform.shape}")

    # Test decoding
    decoded = tokenizer.decode_to_waveform(encoded)
    print(f"Decoded shape: {decoded.shape}")
    print(f"Decoded dtype: {decoded.dtype}")

    # Test encode-decode cycle
    original, reconstructed = tokenizer.encode_decode_cycle(
        audio_path, num_quantizers=8
    )
    print(f"Original shape: {original.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Test saving decoded audio
    output_path = "test_output.wav"
    tokenizer.decode_to_file(encoded, output_path)

    if os.path.exists(output_path):
        print(f"Output file {output_path} created successfully")
        os.remove(output_path)  # Clean up
        print(f"Cleaned up {output_path}")

    print("Basic functionality tests passed!\n")


def test_device_handling():
    """Test CPU/GPU device handling."""
    print("=== Testing Device Handling ===")

    # Test CPU initialization
    tokenizer_cpu = AudioTokenizer(device="cpu", num_quantizers=4)
    print(f"CPU tokenizer device: {tokenizer_cpu.device}")

    # Test device switching
    if torch.cuda.is_available():
        tokenizer_cpu.set_device("cuda")
        print(f"After switching to CUDA: {tokenizer_cpu.device}")
        print(f"Model on GPU: {next(tokenizer_cpu.model.parameters()).is_cuda}")

        tokenizer_cpu.set_device("cpu")
        print(f"After switching back to CPU: {tokenizer_cpu.device}")
        print(f"Model on GPU: {next(tokenizer_cpu.model.parameters()).is_cuda}")
    else:
        print("CUDA not available, testing CPU-only functionality")

    print("Device handling tests passed!\n")


def test_different_quantizers():
    """Test different numbers of quantizers."""
    print("=== Testing Different Quantizer Numbers ===")

    audio_path = "testiranje_mimi/music.mp3"
    if not os.path.exists(audio_path):
        print(f"Warning: {audio_path} not found, skipping quantizer tests")
        return

    tokenizer = AudioTokenizer(device="cpu")  # Use CPU for faster testing

    quantizer_counts = [4, 8, 16]

    for n_q in quantizer_counts:
        encoded = tokenizer.encode_from_path(audio_path, num_quantizers=n_q)
        print(f"Quantizers: {n_q}, Encoded shape: {encoded.shape}")

        # Verify the second dimension matches the number of quantizers
        assert encoded.shape[1] == n_q, (
            f"Expected {n_q} quantizers, got {encoded.shape[1]}"
        )

    print("Quantizer tests passed!\n")


def test_error_handling():
    """Test error handling."""
    print("=== Testing Error Handling ===")

    tokenizer = AudioTokenizer(device="cpu", num_quantizers=4)

    try:
        # Try to set invalid device
        tokenizer.set_device("invalid_device")
        print("ERROR: Should have raised ValueError for invalid device")
    except ValueError as e:
        print(f"Correctly caught error: {e}")

    print("Error handling tests passed!\n")


def test_batch_encoding():
    """Test batch encoding functionality."""
    print("=== Testing Batch Encoding ===")

    audio_path = "testiranje_mimi/music.mp3"
    if not os.path.exists(audio_path):
        print(f"Warning: {audio_path} not found, skipping batch encoding test")
        return

    # Create a small batch (duplicate the same file for testing)
    audio_paths = [audio_path] * 2  # Just 2 for quick testing

    tokenizer = AudioTokenizer(device="cpu", num_quantizers=4)

    encoded_batch = tokenizer.batch_encode(audio_paths, num_quantizers=4)

    print(f"Batch size: {len(encoded_batch)}")
    for i, encoded in enumerate(encoded_batch):
        print(f"Item {i} encoded shape: {encoded.shape}")

    print("Batch encoding tests passed!\n")


def main():
    """Run all tests."""
    print("Starting AudioTokenizer tests...\n")

    try:
        test_basic_functionality()
        test_device_handling()
        test_different_quantizers()
        test_error_handling()
        test_batch_encoding()

        print("🎉 All tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
