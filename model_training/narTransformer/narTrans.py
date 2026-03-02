"""
Architecture:
- Input: Past audio tokens (RVQ encoded)
- Output: Future audio tokens (multiple RVQ codebooks)
- Attention: Block-Causal (Past=Causal, Future=Dense, Past↛Future)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


# Positional Embeddings (RoPE)


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE)
    Encodes relative position information in the attention mechanism.
    Essential for NAR generation where future tokens attend densely.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, dim)
            seq_len: Actual sequence length
        Returns:
            Rotated embeddings
        """
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Apply rotation
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for RoPE rotation"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Attention Mask


def create_block_causal_mask(
    past_len: int, future_len: int, device: torch.device
) -> torch.Tensor:
    """
    Creates the Block-Causal Attention Mask for audio forecasting.

    Structure:
    ┌─────────────┬─────────────┐
    │  Causal     │  MASKED     │  ← Past tokens
    │  (Past→Past)│  (Past↛Future)│
    ├─────────────┼─────────────┤
    │  FULL       │  FULL       │  ← Future tokens
    │  (Future→Past)│  (Future→Future)│
    └─────────────┴─────────────┘

    Args:
        past_len: Number of past context tokens
        future_len: Number of future prediction tokens
        device: torch device

    Returns:
        Attention mask of shape (past_len + future_len, past_len + future_len)
        Values are 0 (attend) or -inf (mask)
    """
    total_len = past_len + future_len
    mask = torch.zeros((total_len, total_len), device=device)

    # 1. Past cannot see Future (Top-Right block)
    mask[:past_len, past_len:] = float("-inf")

    # 2. Past sees Past causally (Top-Left block, lower triangular)
    past_mask = torch.triu(torch.ones(past_len, past_len, device=device), diagonal=1)
    past_mask[past_mask == 1] = float("-inf")
    mask[:past_len, :past_len] = past_mask

    # 3. Future sees Past (Bottom-Left) - Already 0s (full attention)
    # 4. Future sees Future (Bottom-Right) - Already 0s (full attention for NAR)

    return mask


# Transformer Block


class AudioTransformerBlock(nn.Module):
    """
    Standard Transformer Decoder Block with RoPE support.
    Uses pre-normalization (LayerNorm before attention/FFN) for stability.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)

        if cos is not None and sin is not None:
            # Manual attention with RoPE - need to split heads first
            batch, seq_len, _ = x.shape
            head_dim = self.d_model // self.attention.num_heads

            # Project to Q, K, V
            q = k = v = x

            # Reshape for multi-head: (batch, seq_len, n_heads, head_dim)
            q = q.view(batch, seq_len, self.attention.num_heads, head_dim)
            k = k.view(batch, seq_len, self.attention.num_heads, head_dim)
            v = v.view(batch, seq_len, self.attention.num_heads, head_dim)

            # Apply RoPE to each head: cos/sin shape is (1, seq_len, 1, head_dim)
            q, k = apply_rope(q, k, cos, sin)

            # Reshape back to (batch, seq_len, d_model) for PyTorch MHA
            q = q.reshape(batch, seq_len, -1)
            k = k.reshape(batch, seq_len, -1)
            v = v.reshape(batch, seq_len, -1)

            attn_out, _ = self.attention(q, k, v, attn_mask=mask, need_weights=False)
        else:
            attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)

        x = residual + self.dropout(attn_out)

        # Feed-Forward (unchanged)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


# Main Model


class AudioForecaster(nn.Module):
    """
    Hierarchical Non-Autoregressive Audio Forecasting Transformer.

    Predicts multiple RVQ codebooks in parallel for a future time horizon.
    Uses Block-Causal attention for efficient real-time inference.

    Args:
        vocab_size: Size of token vocabulary (DAC codebook size, typically 1024)
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        d_ff: Feed-forward dimension
        n_codebooks: Number of RVQ codebooks to predict
        past_len: Length of past context (in tokens)
        future_len: Length of future prediction (in tokens)
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        n_codebooks: int = 8,
        past_len: int = 250,  # ~5 seconds at 50Hz
        future_len: int = 500,  # ~10 seconds at 50Hz
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        self.past_len = past_len
        self.future_len = future_len
        self.device = device

        # Token Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.codebook_embedding = nn.Embedding(n_codebooks, d_model)

        # Positional Embeddings (RoPE)
        self.rope = RotaryEmbedding(
            dim=d_model // n_heads, max_seq_len=past_len + future_len
        )

        # Transformer Blocks
        self.layers = nn.ModuleList(
            [
                AudioTransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        # Output Heads (one per codebook)
        self.output_heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size) for _ in range(n_codebooks)]
        )

        # Learnable mask token for future prediction
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Register attention mask as buffer (not a parameter)
        self.register_buffer(
            "attn_mask",
            create_block_causal_mask(past_len, future_len, device=torch.device(device)),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed_tokens(
        self, tokens: torch.Tensor, codebook_idx: int = 0
    ) -> torch.Tensor:
        """
        Embed tokens with token + codebook embeddings.

        Args:
            tokens: Token IDs (batch, seq_len)
            codebook_idx: Which codebook these tokens belong to
        Returns:
            Embedded tensor (batch, seq_len, d_model)
        """
        token_emb = self.token_embedding(tokens)
        codebook_emb = self.codebook_embedding(
            torch.tensor(codebook_idx, device=tokens.device)
        )
        return token_emb + codebook_emb

    def forward(
        self, past_tokens: torch.Tensor, return_logits: bool = True
    ) -> List[torch.Tensor]:
        """
        Forward pass for audio forecasting.

        Args:
            past_tokens: Past audio tokens, shape (batch, past_len, n_codebooks)
            return_logits: If True, return logits. If False, return probabilities.

        Returns:
            List of tensors, one per codebook, each shape (batch, future_len, vocab_size)
        """
        batch_size = past_tokens.shape[0]
        total_len = self.past_len + self.future_len

        # =====================================================================
        # 1. Embed Past Tokens
        # =====================================================================
        # past_tokens: (batch, past_len, n_codebooks)
        # We sum embeddings across codebooks for the past context
        past_embeds = torch.zeros(
            batch_size, self.past_len, self.d_model, device=past_tokens.device
        )

        for cb in range(self.n_codebooks):
            past_embeds += self._embed_tokens(past_tokens[:, :, cb], cb)

        # =====================================================================
        # 2. Create Future Mask Tokens
        # =====================================================================
        # Expand mask token for future length
        mask_embeds = self.mask_token.expand(batch_size, self.future_len, self.d_model)

        # =====================================================================
        # 3. Concatenate Past + Future
        # =====================================================================
        x = torch.cat([past_embeds, mask_embeds], dim=1)  # (batch, total_len, d_model)

        # =====================================================================
        # 4. Compute RoPE Embeddings
        # =====================================================================
        cos, sin = self.rope(x, total_len)

        # =====================================================================
        # 5. Transformer Forward Pass
        # =====================================================================
        for layer in self.layers:
            x = layer(x, mask=self.attn_mask, cos=cos, sin=sin)

        x = self.final_norm(x)

        # =====================================================================
        # 6. Extract Future Predictions
        # =====================================================================
        future_hidden = x[:, self.past_len :, :]  # (batch, future_len, d_model)

        # =====================================================================
        # 7. Project to Output Logits (one head per codebook)
        # =====================================================================
        if return_logits:
            logits = [head(future_hidden) for head in self.output_heads]
            return logits  # List of (batch, future_len, vocab_size)
        else:
            probs = [
                torch.softmax(head(future_hidden), dim=-1) for head in self.output_heads
            ]
            return probs

    def predict(
        self,
        past_tokens: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Inference method that returns predicted token IDs.

        Args:
            past_tokens: Past audio tokens (batch, past_len, n_codebooks)
            temperature: Sampling temperature (>1 = more random, <1 = more confident)
            top_k: If set, only sample from top-k tokens

        Returns:
            Predicted tokens (batch, future_len, n_codebooks)
        """
        self.eval()
        batch_size = past_tokens.shape[0]
        with torch.no_grad():
            logits = self.forward(past_tokens, return_logits=True)

            predictions = []
            for cb_logits in logits:
                # Apply temperature
                cb_logits = cb_logits / temperature

                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = (
                        cb_logits < torch.topk(cb_logits, top_k)[0][..., -1, None]
                    )
                    cb_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = torch.softmax(cb_logits, dim=-1)
                predicted_tokens = torch.multinomial(
                    probs.view(-1, self.vocab_size), num_samples=1
                ).view(batch_size, self.future_len)

                predictions.append(predicted_tokens)

            # Stack codebooks
            return torch.stack(predictions, dim=-1)  # (batch, future_len, n_codebooks)

    def get_training_loss(
        self,
        past_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        fidelity_decay: bool = True,
    ) -> torch.Tensor:
        """
        Compute training loss with optional fidelity decay.

        Fidelity decay reduces loss weight for higher codebooks at distant time steps.
        This teaches the model that fine details are uncertain far in the future.

        Args:
            past_tokens: (batch, past_len, n_codebooks)
            future_tokens: (batch, future_len, n_codebooks) - ground truth
            fidelity_decay: If True, apply time-based loss weighting

        Returns:
            Scalar loss tensor
        """
        logits = self.forward(past_tokens, return_logits=True)

        total_loss = 0.0
        loss_fn = nn.CrossEntropyLoss()

        for cb_idx, cb_logits in enumerate(logits):
            # Reshape for loss: (batch * future_len, vocab_size)
            cb_logits_flat = cb_logits.view(-1, self.vocab_size)
            cb_targets_flat = future_tokens[:, :, cb_idx].reshape(-1)

            # Compute base loss
            cb_loss = loss_fn(cb_logits_flat, cb_targets_flat)

            # Apply fidelity decay weight
            if fidelity_decay:
                # Weight decreases for higher codebooks and later time steps
                cb_weight = 1.0 / (cb_idx + 1)  # CB0=1.0, CB1=0.5, CB2=0.33...
                total_loss += cb_weight * cb_loss
            else:
                total_loss += cb_loss

        return total_loss / self.n_codebooks
