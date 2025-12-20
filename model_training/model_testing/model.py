import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding from "Attention Is All You Need".

    Args:
        d_model (int): The expected feature size of the input embeddings
        max_len (int): Maximum sequence length the model can handle
        dropout (float): Dropout probability (default: 0.1)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create buffer for positional encodings (max_len x d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices [0, 1, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # Shape: (max_len, 1)

        # Compute div_term for frequency scaling
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  # Shape: (d_model//2,)

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices (0, 2, 4, ...)
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices (1, 3, 5, ...)

        # Register as buffer (not a parameter, but part of state dict)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor with positional encodings added (batch_size, seq_len, d_model)
        """
        # Add positional encoding up to the sequence length of the input
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TestModel(nn.Module):
    def __init__(
        self,
        history_length: int = 5,  # Historical RVQ tensors to use
        future_length: int = 2,  # Future RVQ tensors to predict
        rvq_levels: int = 12,  # RVQ codebook levels
        conv_channels: int = 128,
        embedding_dim: int = 128,
        num_heads: int = 8,  # Fixed: was 128, typically 8 or 16
        trans_layers: int = 2,  # Fixed: was 128, typically 2-12
        codebook_size: int = 1024,
    ):
        super(TestModel, self).__init__()

        # Validate embedding dimension compatibility
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dim {embedding_dim} must be divisible by num_heads {num_heads}"
            )
        self.history_length = history_length
        self.future_length = future_length
        self.rvq_levels = rvq_levels
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        # 1D convolutional pass - process flattened input
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)

        # Adaptive pooling to ensure consistent output size regardless of input temporal length
        self.temporal_pool = nn.AdaptiveAvgPool1d(history_length * rvq_levels)

        # Calculate the flattened size after convolutions and pooling
        conv_output_length = history_length * rvq_levels

        # Linear layer to project conv features to desired size
        self.conv_linear = nn.Linear(
            in_features=conv_channels * conv_output_length,
            out_features=history_length * embedding_dim,
        )

        # positional encoding
        self.positional = PositionalEncoding(
            d_model=embedding_dim,
            max_len=history_length + future_length,  # Extended for future predictions
            dropout=0.1,
        )

        # linear projection for transformer input
        self.linear_projection = nn.Linear(rvq_levels, embedding_dim)

        # Transformer encoder for processing historical context
        encoder_layer = nn.TransformerEncoderLayer(
            nhead=num_heads,
            d_model=embedding_dim,
            dim_feedforward=embedding_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=trans_layers,
            norm=nn.LayerNorm(embedding_dim),
        )

        # Project to future sequence length using learned projection
        self.future_projector = nn.Linear(
            history_length * embedding_dim, future_length * embedding_dim
        )

        # Output layers for predicting future RVQ tokens
        self.output_projections = nn.ModuleList(
            [nn.Linear(embedding_dim, codebook_size) for _ in range(rvq_levels)]
        )

    def forward(self, historical_rvq: torch.Tensor):
        # Convert input to float if it's not already
        historical_rvq = historical_rvq.float()
        (B, temporal_length, RVQ) = historical_rvq.shape

        # Conv processing path - process the flattened temporal sequence
        conv_in = historical_rvq.reshape((B, 1, -1))  # -> B, 1, (temporal_length * RVQ)
        conv_out = torch.relu(
            self.conv1(conv_in)
        )  # -> B, conv_channels, (temporal_length * RVQ)
        conv_out = torch.relu(
            self.conv2(conv_out)
        )  # -> B, conv_channels, (temporal_length * RVQ)

        # Apply adaptive pooling to ensure consistent temporal dimension
        # Pool to the expected temporal length (history_length * rvq_levels)
        expected_temporal_length = self.history_length * RVQ
        if conv_out.size(2) != expected_temporal_length:
            conv_out = self.temporal_pool(
                conv_out
            )  # -> B, conv_channels, (history_length * rvq_levels)

        conv_out = conv_out.view(
            B, -1
        )  # Flatten to (B, conv_channels * history_length * rvq_levels)
        conv_out = self.conv_linear(conv_out)  # -> B, (history_length * embedding_dim)
        conv_out = conv_out.reshape((B, self.history_length, self.embedding_dim))

        # Transformer processing path - need to handle variable temporal length
        # If temporal_length != history_length, we need to adjust the transformer path
        if temporal_length != self.history_length:
            # Use adaptive pooling to reduce temporal dimension to history_length
            # Reshape: [B, temporal_length, RVQ] -> [B, RVQ, temporal_length]
            # Then pool along temporal dimension to get history_length
            reshaped_rvq = historical_rvq.transpose(
                1, 2
            )  # -> [B, RVQ, temporal_length]
            pooled_rvq = nn.functional.adaptive_avg_pool1d(
                reshaped_rvq, self.history_length
            )  # -> [B, RVQ, history_length]
            pooled_historical_rvq = pooled_rvq.transpose(
                1, 2
            )  # -> [B, history_length, RVQ]
        else:
            pooled_historical_rvq = historical_rvq

        trans_out = self.linear_projection(
            pooled_historical_rvq
        )  # B, history_length, embedding_dim
        trans_out = self.transformer_encoder(trans_out)  # Process historical context

        # Combine both paths
        combined_context = trans_out + conv_out  # Residual connection

        # Project to future sequence length
        future_context = self.future_projector(combined_context.reshape(B, -1))
        future_context = future_context.reshape(
            B, self.future_length, self.embedding_dim
        )

        # Apply positional encoding to future context
        future_context = self.positional(future_context)

        # Generate logits for each RVQ level independently
        future_logits = []
        for i, output_proj in enumerate(self.output_projections):
            # Apply the same transformation to all timesteps
            level_logits = output_proj(
                future_context
            )  # B, future_length, codebook_size
            future_logits.append(
                level_logits.unsqueeze(-1)
            )  # B, future_length, codebook_size, 1

        # Stack along RVQ levels dimension
        future_logits = torch.cat(
            future_logits, dim=-1
        )  # B, future_length, codebook_size, rvq_levels

        # Transpose to get final shape: B, future_length, rvq_levels, codebook_size
        future_logits = future_logits.permute(
            0, 1, 3, 2
        )  # B, future_length, rvq_levels, codebook_size

        return future_logits

    def sample_future_tokens(
        self, historical_rvq: torch.Tensor, temperature: float = 1.0
    ):
        """Sample future RVQ tokens given historical context."""
        logits = self.forward(
            historical_rvq
        )  # B, future_length, rvq_levels, codebook_size

        # Apply temperature scaling
        scaled_logits = logits / temperature

        # Apply softmax to get probabilities
        probs = torch.softmax(
            scaled_logits, dim=-1
        )  # B, future_length, rvq_levels, codebook_size

        # Sample from the distribution
        sampled_tokens = torch.multinomial(
            probs.view(-1, self.codebook_size), num_samples=1
        ).view(logits.shape[:-1])  # B, future_length, rvq_levels

        return sampled_tokens


if __name__ == "__main__":
    from torchinfo import summary

    # Create model instance with typical parameters
    model = TestModel(history_length=5, future_length=2, rvq_levels=12)
    print("TestModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Test forward pass with correct dimensions
    historical_rvq = torch.randint(
        0, 1024, (1, 5, 12)
    )  # (batch_size=1, history_length=5, rvq_levels=12)
    output_logits = model(historical_rvq)
    print(f"\nForward pass test - Input: {historical_rvq.shape}")
    print(f"Output logits shape: {output_logits.shape}")  # Should be [1, 2, 12, 1024]

    # Test sampling
    sampled_tokens = model.sample_future_tokens(historical_rvq)
    print(f"Sampled tokens shape: {sampled_tokens.shape}")  # Should be [1, 2, 12]

    # Summary with proper input dimensions - convert to float for the summary
    dummy_input = torch.randint(
        0, 1024, (1, 5, 12), dtype=torch.float
    )  # Changed to float
    try:
        summary(
            model,
            input_data=(dummy_input,),
            device="cpu",
            verbose=1,
        )
    except Exception as e:
        print(f"Summary failed: {e}")
