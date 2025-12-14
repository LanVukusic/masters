import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.JointModel import JointAudioContinuationModel
from model_training.dataloader.audio_tokenized_dataset import AudioTokenizedDataset
from model_training.tokenizer.mimi_audio_tokenizer import MimiAudioTokenizer
from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# Configuration
config = {
    "history_length": 6,
    "future_frames": 2,
    "rvq_levels": 6,
    "embedding_dim": 256,
    "batch_size": 6,
    "codebook_size": 1024,  # Updated to match Mimi tokenizer range (0-2047) or DAC 1024
    "learning_rate": 3e-4,
    "num_epochs": 1000,
    "audio_dir": "dataset_gen/rotormotor/mp3s",  # Update this path as needed
    "tokenizer_type": "DAC",  # or "MIMI"
    "device": device
}



# Create dataset and dataloader
dataset = AudioTokenizedDataset(
    audio_dir=config["audio_dir"],
    tokenizer=config["tokenizer_type"],
    num_chunks=config["history_length"] + config["future_frames"],  # Total sequence length
    rvq_depth=config["rvq_levels"],
    cache_size=10,  # Keep 20 songs in memory
    preload_cache=True,  # Preload songs on initialization
    device=config["device"]
)

dataloader = DataLoader(
    dataset, 
    batch_size=config["batch_size"], 
    shuffle=True, 
    num_workers=8  # Set to 0 to avoid multiprocessing issues during debugging
)

print(f"Dataset created successfully!")
print(f"Number of audio files found: {len(dataset.audio_files)}")
print(f"Estimated dataset length: {len(dataset)}")
print(f"Dataloader batch size: {config['batch_size']}")



# Initialize model AFTER dataset creation to ensure correct codebook_size
model = JointAudioContinuationModel(
    history_length=config["history_length"],
    future_frames=config["future_frames"],
    rvq_levels=config["rvq_levels"],
    embedding_dim=config["embedding_dim"],
    codebook_size=config["codebook_size"],
)
model.to(device)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/audio_continuation_training")

# Training loop with actual data from dataloader
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

print("Starting training...")
for epoch in range(config["num_epochs"]):
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # batch shape: [batch_size, rvq_levels, time_steps]
        batch_size_current, rvq_levels, total_time_steps = batch.shape
        
        # Split the batch into historical and future parts
        # Assuming each chunk represents one time step
        frames_per_chunk = (dataset.chunk_duration * dataset.sampling_rate) // dataset.samples_per_frame
        
        # We need to split the temporal dimension appropriately
        total_chunks_needed = config["history_length"] + config["future_frames"]
        
        # Calculate how many time steps correspond to our needed chunks
        time_steps_per_chunk = total_time_steps // total_chunks_needed
        
        if time_steps_per_chunk < 1:
            print(f"Warning: Not enough time steps in batch. Skipping batch {batch_idx}")
            continue
            
        # Extract historical and future RVQ tokens
        historical_end = config["history_length"] * time_steps_per_chunk
        future_start = historical_end
        future_end = future_start + config["future_frames"] * time_steps_per_chunk
        
        historical_rvq = batch[:, :, :historical_end] # [batch_size, rvq_levels, history_time_steps]
        future_rvq = batch[:, :, future_start:future_end] # [batch_size, rvq_levels, future_time_steps]
        
        # Rearrange to match model expectations: [batch_size, time_steps, rvq_levels]
        historical_rvq = historical_rvq.transpose(1, 2)  # [batch_size, history_time_steps, rvq_levels]
        future_rvq = future_rvq.transpose(1, 2)  # [batch_size, future_time_steps, rvq_levels]
        
        # Validate token ranges to prevent index out of bounds
        max_token_value = config["codebook_size"] - 1
        if historical_rvq.max() > max_token_value or historical_rvq.min() < 0:
            print(f"Warning: Historical RVQ tokens out of range! Max: {historical_rvq.max()}, Min: {historical_rvq.min()}")
            # Clamp values to valid range
            historical_rvq = torch.clamp(historical_rvq, 0, max_token_value)
        
        if future_rvq.max() > max_token_value or future_rvq.min() < 0:
            print(f"Warning: Future RVQ tokens out of range! Max: {future_rvq.max()}, Min: {future_rvq.min()}")
            # Clamp values to valid range
            future_rvq = torch.clamp(future_rvq, 0, max_token_value)
        
        # Convert to long type (required for embedding lookup)
        historical_rvq = historical_rvq.long()
        future_rvq = future_rvq.long()
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(historical_rvq, future_rvq)
        
        # Backward pass
        loss = output["total_loss"]
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        # Log batch-level metrics every 10 batches
        if batch_idx % 10 == 0:
            writer.add_scalar('Batch/Loss', loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Batch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + batch_idx)
            
            # Log gradient norm
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            writer.add_scalar('Batch/Gradient_Norm', total_grad_norm, epoch * len(dataloader) + batch_idx)
        
        # Print progress for first few batches in each epoch
        print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
    
    if batch_count > 0:
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Average Loss: {avg_epoch_loss:.4f}")
        
        # Log epoch-level metrics
        writer.add_scalar('Epoch/Average_Loss', avg_epoch_loss, epoch)
        writer.add_scalar('Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Log gradient norm at epoch level
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        writer.add_scalar('Epoch/Gradient_Norm', total_grad_norm, epoch)
    else:
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, No valid batches processed")

# # Inference example with dummy data (since we need to test the model's generation capability)
# print("\nRunning inference...")
# model.eval()
# with torch.no_grad():
#     # Create dummy historical data for inference testing
#     dummy_historical = torch.randint(
#         0, config["codebook_size"], 
#         (config["batch_size"], config["history_length"], config["rvq_levels"])
#     )
#     generated = model.generate(dummy_historical)
#     print(f"Generated RVQ shape: {generated.shape}")
#     print(
#         f"Generated RVQ min/max: {generated.min().item()}, {generated.max().item()}"
#     )

# Close TensorBoard writer
writer.close()

# Save model
torch.save(model.state_dict(), "audio_continuation_model.pth")
print("Model saved successfully!")
