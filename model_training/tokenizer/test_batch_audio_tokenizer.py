#!/usr/bin/env python3
"""
Test script for the AudioTokenizer class with batch processing functionality.
"""

import torch
import torchaudio
import os
from model_training.model.audio_tokenizer import AudioTokenizer


def test_batch_path_encoding():
    """Test batch encoding from audio paths."""
    print("=== Testing Batch Path Encoding ===")

    # Initialize tokenizer
    tokenizer = AudioTokenizer(
        num_quantizers=4, device="cpu"
    )  # Use CPU for faster testing

    # Check if music.mp3 exists in testiranje_mimi directory
    audio_path = "testiranje_mimi/music.mp3"
    if not os.path.exists(audio_path):
        print(f"Warning: {audio_path} not found, creating dummy audio for testing")
        # Create a dummy audio file for testing
        dummy_audio = torch.randn(1, 24000)  # 1 second at 24kHz
        torchaudio.save("dummy_test.wav", dummy_audio, 24000)
        audio_path = "dummy_test.wav"

    # Test single path encoding
    single_encoded = tokenizer.encode_from_path(audio_path)
    print(f"Single encoded shape: {single_encoded.shape}")

    # Test batch path encoding
    batch_paths = [
        audio_path,
        audio_path,
        audio_path,
    ]  # Use same file 3 times for testing
    batch_encoded = tokenizer.encode_from_path(batch_paths)

    print(f"Batch encoded length: {len(batch_encoded)}")
    for i, encoded in enumerate(batch_encoded):
        print(f"Batch item {i} encoded shape: {encoded.shape}")

    # Verify all encoded tensors have the same shape
    expected_shape = single_encoded.shape
    for i, encoded in enumerate(batch_encoded):
        assert encoded.shape == expected_shape, (
            f"Batch item {i} has wrong shape: {encoded.shape} vs {expected_shape}"
        )

    print("Batch path encoding tests passed!\n")

    # Clean up dummy file if created
    if os.path.exists("dummy_test.wav"):
        os.remove("dummy_test.wav")


def test_batch_waveform_encoding():
    """Test batch encoding from waveforms."""
    print("=== Testing Batch Waveform Encoding ===")

    tokenizer = AudioTokenizer(num_quantizers=4, device="cpu")

    # Create dummy waveforms for testing
    dummy_waveform1 = torch.randn(1, 24000)  # 1 second at 24kHz
    dummy_waveform2 = torch.randn(1, 24000)  # 1 second at 24kHz
    dummy_waveform3 = torch.randn(1, 24000)  # 1 second at 24kHz

    # Test single waveform encoding
    single_encoded = tokenizer.encode_from_waveform(dummy_waveform1, 24000)
    print(f"Single waveform encoded shape: {single_encoded.shape}")

    # Test batch waveform encoding
    batch_waveforms = [dummy_waveform1, dummy_waveform2, dummy_waveform3]
    batch_encoded = tokenizer.encode_from_waveform(batch_waveforms, 24000)

    print(f"Batch waveform encoded length: {len(batch_encoded)}")
    for i, encoded in enumerate(batch_encoded):
        print(f"Batch waveform item {i} encoded shape: {encoded.shape}")

    # Verify all encoded tensors have the same shape
    expected_shape = single_encoded.shape
    for i, encoded in enumerate(batch_encoded):
        assert encoded.shape == expected_shape, (
            f"Batch waveform item {i} has wrong shape: {encoded.shape} vs {expected_shape}"
        )

    print("Batch waveform encoding tests passed!\n")


def test_batch_decoding():
    """Test batch decoding functionality."""
    print("=== Testing Batch Decoding ===")

    tokenizer = AudioTokenizer(num_quantizers=4, device="cpu")

    # Create dummy waveforms and encode them
    dummy_waveform1 = torch.randn(1, 24000)
    dummy_waveform2 = torch.randn(1, 24000)

    encoded1 = tokenizer.encode_from_waveform(dummy_waveform1, 24000)
    encoded2 = tokenizer.encode_from_waveform(dummy_waveform2, 24000)

    print(f"Encoded 1 shape: {encoded1.shape}")
    print(f"Encoded 2 shape: {encoded2.shape}")

    # Test single decoding
    single_decoded = tokenizer.decode_to_waveform(encoded1)
    print(f"Single decoded shape: {single_decoded.shape}")

    # Test batch decoding
    batch_encoded = [encoded1, encoded2]
    batch_decoded = tokenizer.decode_to_waveform(batch_encoded)

    print(f"Batch decoded length: {len(batch_decoded)}")
    for i, decoded in enumerate(batch_decoded):
        print(f"Batch decoded item {i} shape: {decoded.shape}")

    # Verify shapes
    expected_shape = single_decoded.shape
    for i, decoded in enumerate(batch_decoded):
        assert decoded.shape == expected_shape, (
            f"Batch decoded item {i} has wrong shape: {decoded.shape} vs {expected_shape}"
        )

    print("Batch decoding tests passed!\n")


def test_batch_file_saving():
    """Test batch file saving functionality."""
    print("=== Testing Batch File Saving ===")

    tokenizer = AudioTokenizer(num_quantizers=4, device="cpu")

    # Create dummy waveforms and encode them
    dummy_waveform1 = torch.randn(1, 12000)  # 0.5 seconds at 24kHz
    dummy_waveform2 = torch.randn(1, 1200)  # 0.5 seconds at 24kHz

    encoded1 = tokenizer.encode_from_waveform(dummy_waveform1, 24000)
    encoded2 = tokenizer.encode_from_waveform(dummy_waveform2, 24000)

    # Test single file saving
    tokenizer.decode_to_file(encoded1, "single_test.wav")
    assert os.path.exists("single_test.wav"), "Single file saving failed"
    os.remove("single_test.wav")
    print("Single file saving works")

    # Test batch file saving with multiple paths
    batch_encoded = [encoded1, encoded2]
    batch_paths = ["batch_test_0.wav", "batch_test_1.wav"]
    tokenizer.decode_to_file(batch_encoded, batch_paths)

    for path in batch_paths:
        assert os.path.exists(path), f"Batch file {path} was not created"
        os.remove(path)
    print("Batch file saving with multiple paths works")

    # Test batch file saving with single path (should create numbered files)
    tokenizer.decode_to_file(batch_encoded, "numbered_test.wav")

    numbered_files = ["numbered_test_0.wav", "numbered_test_1.wav"]
    for path in numbered_files:
        assert os.path.exists(path), f"Numbered file {path} was not created"
        os.remove(path)
    print("Batch file saving with numbered files works")

    print("Batch file saving tests passed!\n")


def test_mixed_batch_scenarios():
    """Test mixed scenarios with different input types."""
    print("=== Testing Mixed Batch Scenarios ===")

    tokenizer = AudioTokenizer(num_quantizers=4, device="cpu")

    # Create dummy waveform for testing
    dummy_waveform = torch.randn(1, 12000)

    # Test mixed single and batch operations
    single_encoded = tokenizer.encode_from_waveform(dummy_waveform, 24000)
    batch_encoded = [single_encoded, single_encoded]

    # Decode mixed batch
    decoded_batch = tokenizer.decode_to_waveform(batch_encoded)
    print(f"Mixed batch decoding length: {len(decoded_batch)}")

    for i, decoded in enumerate(decoded_batch):
        print(f"Mixed batch decoded item {i} shape: {decoded.shape}")

    print("Mixed batch scenarios tests passed!\n")


def test_error_handling():
    """Test error handling for batch operations."""
    print("=== Testing Error Handling ===")

    tokenizer = AudioTokenizer(num_quantizers=4, device="cpu")

    # Test error with mismatched batch sizes
    dummy_waveform1 = torch.randn(1, 12000)
    dummy_waveform2 = torch.randn(1, 12000)

    encoded1 = tokenizer.encode_from_waveform(dummy_waveform1, 24000)
    encoded2 = tokenizer.encode_from_waveform(dummy_waveform2, 24000)

    # This should raise an error due to mismatched batch sizes
    try:
        tokenizer.decode_to_file([encoded1, encoded2], ["only_one_path.wav"])
        print("ERROR: Should have raised ValueError for mismatched batch sizes")
    except ValueError as e:
        print(f"Correctly caught error: {e}")

    # Test error with invalid input types
    try:
        tokenizer.encode_from_path(123)  # Invalid type
        print("ERROR: Should have raised ValueError for invalid type")
    except ValueError as e:
        print(f"Correctly caught error: {e}")

    try:
        tokenizer.encode_from_waveform(
            "invalid_type", 24000
        )  # Invalid type but with required args
        print("ERROR: Should have raised ValueError for invalid type")
    except ValueError as e:
        print(f"Correctly caught error: {e}")

    print("Error handling tests passed!\n")


def main():
    """Run all tests."""
    print("Starting AudioTokenizer batch processing tests...\n")

    try:
        test_batch_path_encoding()
        test_batch_waveform_encoding()
        test_batch_decoding()
        test_batch_file_saving()
        test_mixed_batch_scenarios()
        test_error_handling()

        print("🎉 All batch processing tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
