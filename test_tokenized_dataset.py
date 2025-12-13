from torch.utils.data import DataLoader
from model_training.dataloader.audio_tokenized_dataset import (
    AudioTokenizedDataset,
)
from model_training.tokenizer.audio_tokenizer import AudioTokenizer


def test_dataset():
    """
    Test the AudioTokenizedDataset with a directory of MP3 files.
    Make sure to update the audio_dir path to point to your actual MP3 files.
    """

    # Update this path to point to your directory containing .mp3 files
    audio_dir = "dataset_gen/rotormotor/mp3s"  # Adjust this path as needed

    # Create tokenizer
    tokenizer = AudioTokenizer(
        num_quantizers=8,  # Using 8 quantizers as specified
        device="cpu",  # Use "cuda" if you have GPU
    )

    print(f"Tokenizer sampling rate: {tokenizer.sampling_rate}")
    print(f"Tokenizer frame size: {tokenizer.frame_size}")

    # Create dataset
    try:
        dataset = AudioTokenizedDataset(
            audio_dir=audio_dir,
            tokenizer=tokenizer,
            num_chunks=4,  # 4 chunks per sample
            rvq_depth=8,  # 8 quantizers
            chunk_duration=2.0,  # 2 seconds per chunk
            max_samples_per_file=5,  # Max 5 samples per file
        )

        print("Dataset created successfully!")
        print(f"Number of audio files found: {len(dataset.audio_files)}")
        print(f"Estimated dataset length: {len(dataset)}")
        print(f"Chunk duration: {dataset.chunk_duration}s")
        print(f"Chunk size in samples: {dataset.chunk_size_samples}")
        print(f"Total sequence samples: {dataset.total_sequence_samples}")
        print(f"Samples per frame: {dataset.samples_per_frame}")

        # Test getting a single item
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample shape: {sample.shape}")
            print("Expected shape: [rvq_depth, time_steps] = [8, ?]")

            # The time_steps should be related to the total sequence length divided by frame processing
            expected_time_steps = (
                dataset.total_sequence_samples // dataset.samples_per_frame
            )
            print(f"Expected time steps: {expected_time_steps}")

    except ValueError as e:
        print(f"Error creating dataset: {e}")
        print("Make sure the audio_dir contains .mp3 files")
        return

    # Test with dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    print("\nTesting with dataloader:")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1} shape: {batch.shape}")
        print(f"Batch dtype: {batch.dtype}")

        # Check if we have the expected dimensions
        batch_size, rvq_depth, time_steps = batch.shape
        print(f"  - Batch size: {batch_size}")
        print(f"  - RVQ depth: {rvq_depth}")
        print(f" - Time steps: {time_steps}")

        if i >= 2:  # Test a few batches
            break


if __name__ == "__main__":
    print("Testing AudioTokenizedDataset...")
    test_dataset()
