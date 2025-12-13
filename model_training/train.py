import torch
from model.JointModel import JointAudioContinuationModel

# Example usage and training setup
if __name__ == "__main__":
    # Configuration
    config = {
        "history_length": 5,
        "future_frames": 2,
        "rvq_levels": 12,
        "embedding_dim": 512,
        "batch_size": 4,
        "codebook_size": 1024,
        "learning_rate": 3e-4,
        "num_epochs": 10,
    }

    # Initialize model
    model = JointAudioContinuationModel(
        history_length=config["history_length"],
        future_frames=config["future_frames"],
        rvq_levels=config["rvq_levels"],
        embedding_dim=config["embedding_dim"],
        codebook_size=config["codebook_size"],
    )

    # Example dummy data
    batch_size = config["batch_size"]
    history_length = config["history_length"]
    future_frames = config["future_frames"]
    rvq_levels = config["rvq_levels"]
    codebook_size = config["codebook_size"]

    # Random historical RVQ tokens (simulating real data)
    historical_rvq = torch.randint(
        0, codebook_size, (batch_size, history_length, rvq_levels)
    )
    future_rvq = torch.randint(
        0, codebook_size, (batch_size, future_frames, rvq_levels)
    )

    # Training loop example
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    print("Starting training...")
    for epoch in range(config["num_epochs"]):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(historical_rvq, future_rvq)

        # Backward pass
        loss = output["total_loss"]
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {loss.item():.4f}")

    # Inference example
    print("\nRunning inference...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(historical_rvq)
        print(f"Generated RVQ shape: {generated.shape}")
        print(
            f"Generated RVQ min/max: {generated.min().item()}, {generated.max().item()}"
        )

    # Save model
    torch.save(model.state_dict(), "audio_continuation_model.pth")
    print("Model saved successfully!")
