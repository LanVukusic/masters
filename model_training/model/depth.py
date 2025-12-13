import torch
import torch.nn as nn
from typing import Tuple, List, Optional

class DepthModel(nn.Module):
    def __init__(
        self,
        rvq_levels: int = 12,
        embedding_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        codebook_size: int = 1024,
    ):
        super().__init__()
        self.rvq_levels = rvq_levels
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size

        # Frame token to initial hidden state
        self.frame_to_hidden = nn.Linear(embedding_dim, embedding_dim)

        # Autoregressive transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embedding_dim)
        )

        # Output projection for each RVQ level
        self.output_heads = nn.ModuleList(
            [nn.Linear(embedding_dim, codebook_size) for _ in range(rvq_levels)]
        )

        # Learnable start tokens for each RVQ level
        self.start_tokens = nn.Parameter(torch.randn(rvq_levels, embedding_dim))

        # Embedding layers for RVQ tokens
        self.rvq_embeddings = nn.ModuleList(
            [nn.Embedding(codebook_size, embedding_dim) for _ in range(rvq_levels)]
        )

    def forward(
        self, frame_token: torch.Tensor, target_rvq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = frame_token.shape[0]

        # Initialize hidden state from frame token
        hidden_state = self.frame_to_hidden(frame_token)  # (batch_size, embedding_dim)
        memory = hidden_state.unsqueeze(1)  # (batch_size, 1, embedding_dim)

        # Track generated token embeddings for previous levels
        generated_embeddings = []  # Will hold embeddings for levels 0 to current-1
        logits_list = []
        generated_indices = []

        for level in range(self.rvq_levels):
            # Build input sequence: [generated tokens for prev levels] + [start token for current level]
            if level == 0:
                # First level: only start token
                input_sequence = self.start_tokens[0].unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
            else:
                # Concatenate previously generated embeddings (levels 0 to level-1)
                prev_tokens = torch.stack(generated_embeddings, dim=1)  # (batch_size, level, embedding_dim)
                # Add start token for current level
                current_start = self.start_tokens[level].unsqueeze(0).expand(batch_size, -1).unsqueeze(1)
                input_sequence = torch.cat([prev_tokens, current_start], dim=1)  # (batch_size, level+1, embedding_dim)

            # Create causal mask
            tgt_mask = self._generate_square_subsequent_mask(input_sequence.size(1)).to(frame_token.device)
            
            # Transformer pass
            transformer_output = self.transformer(
                input_sequence, memory, tgt_mask=tgt_mask
            )  # (batch_size, seq_len, embedding_dim)
            
            # Get last position output (current level prediction)
            last_output = transformer_output[:, -1]  # (batch_size, embedding_dim)
            
            # Get logits for current level
            logits = self.output_heads[level](last_output)  # (batch_size, codebook_size)
            logits_list.append(logits)
            
            # Get next token
            if self.training and target_rvq is not None:
                next_token_idx = target_rvq[:, level]  # Teacher forcing
            else:
                next_token_idx = torch.argmax(logits, dim=-1)  # Inference
            
            generated_indices.append(next_token_idx)
            
            # Get embedding for generated token (for next levels)
            next_token_embed = self.rvq_embeddings[level](next_token_idx)  # (batch_size, embedding_dim)
            generated_embeddings.append(next_token_embed)

        generated_rvq = torch.stack(generated_indices, dim=1)  # (batch_size, rvq_levels)
        return generated_rvq, logits_list

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        return mask.bool()


if __name__ == "__main__":
    from torchinfo import summary
    
    # Create model instance
    model = DepthModel(
        rvq_levels=12,
        embedding_dim=512,
        num_layers=2,
        num_heads=4,
        codebook_size=1024
    )
    
    print("DepthModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass with proper output handling
    frame_token = torch.randn(2, 512)  # batch_size=2 for better testing
    generated_rvq, logits_list = model(frame_token)
    
    print("\nForward pass test results:")
    print(f"Input shape: {frame_token.shape}")
    print(f"Generated RVQ shape: {generated_rvq.shape} (expected: [batch_size, rvq_levels])")
    print(f"Logits list length: {len(logits_list)} (one per RVQ level)")
    print(f"Each logits tensor shape: {logits_list[0].shape} (expected: [batch_size, codebook_size])")
    print(f"First logits values (batch 0): {logits_list[0][0, :5]}")  # Show first 5 values
    
    # Optional: Verify gradients flow correctly
    loss = sum(logits.sum() for logits in logits_list)  # Dummy loss
    loss.backward()
    print("\nGradient check: All parameters received gradients? ", 
          all(p.grad is not None for p in model.parameters() if p.requires_grad))
