import torch
import torch.nn as nn


class FrameModel(nn.Module):
    """
    Frame Model: Bidirectional encoder that processes historical RVQ tokens
    and outputs frame tokens for future prediction.

    Based on patent sections [00028], [00044]:
    - Uses bidirectional transformer architecture
    - Processes all historical context at once
    - Outputs N frame tokens for next N frames
    - Computational complexity O(T²) instead of O(T²×Q²)
    """

    def __init__(
        self,
        rvq_levels: int = 12,  # Number of RVQ codebook levels
        embedding_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        future_frames: int = 2,  # Number of future frames to predict
    ):
        super().__init__()
        self.rvq_levels = rvq_levels
        self.embedding_dim = embedding_dim
        self.future_frames = future_frames

        # Embedding layer for RVQ tokens (each level gets its own embedding)
        self.rvq_embeddings = nn.ModuleList(
            [
                nn.Embedding(1024, embedding_dim)
                for _ in range(rvq_levels)  # Assuming 1024 codebook size
            ]
        )

        # Bidirectional transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embedding_dim)
        )

        # Output head to predict future frame tokens
        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim * future_frames),
        )

    def forward(self, historical_rvq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            historical_rvq: (batch_size, history_length, rvq_levels)
                Historical RVQ tokens, where history_length=5, rvq_levels=12

        Returns:
            future_frame_tokens: (batch_size, future_frames, embedding_dim)
                Frame tokens for the next `future_frames` frames
        """
        batch_size, history_length, rvq_levels = historical_rvq.shape

        # Embed each RVQ level separately and average
        embedded_tokens = []
        for level in range(rvq_levels):
            level_tokens = historical_rvq[:, :, level]  # (batch_size, history_length)
            embedded = self.rvq_embeddings[level](
                level_tokens
            )  # (batch_size, history_length, embedding_dim)
            embedded_tokens.append(embedded)

        # Average across RVQ levels to get frame-level representation
        frame_representations = torch.stack(embedded_tokens, dim=2).mean(
            dim=2
        )  # (batch_size, history_length, embedding_dim)

        # Apply bidirectional transformer
        transformer_output = self.transformer(
            frame_representations
        )  # (batch_size, history_length, embedding_dim)

        # Use the last frame's representation to predict future frames
        last_frame_repr = transformer_output[:, -1]  # (batch_size, embedding_dim)

        # Predict future frame tokens
        future_tokens_flat = self.output_head(
            last_frame_repr
        )  # (batch_size, embedding_dim * future_frames)
        future_frame_tokens = future_tokens_flat.view(
            batch_size, self.future_frames, self.embedding_dim
        )

        return future_frame_tokens


if __name__ == "__main__":
    from torchinfo import summary
    
    # Create model instance with typical parameters
    model = FrameModel(
        rvq_levels=12,
        embedding_dim=256,
        num_layers=3,
        num_heads=4,
        future_frames=2
    )
    
    print("FrameModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    historical_rvq = torch.randint(0, 1024, (1, 5, 12))  # (batch_size=1, history_length=5, rvq_levels=12)
    output = model(historical_rvq)
    print(f"\nForward pass test - Input: {historical_rvq.shape}, Output: {output.shape}")
    
    # FIX: Use correct input_size format for torchinfo with batch dimension and dtype
    summary(
        model,
        input_size=(1, 5, 12),  # Include batch dimension
        device='cpu',
        dtypes=[torch.long]  # Specify dtype for embedding layers
    )
