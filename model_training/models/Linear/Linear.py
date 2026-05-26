import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from fla.layers import GatedDeltaNet
from fla.modules import RMSNorm


class LinearDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = RMSNorm(d_model, eps=1e-6)
        self.norm2 = RMSNorm(d_model, eps=1e-6)

        self.gated_delta_net = GatedDeltaNet(
            hidden_size=d_model,
            num_heads=num_heads,
            head_dim=self.head_dim,
            use_gate=True,
            use_short_conv=True,
            mode="chunk",
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.norm1(x)
        x = self.gated_delta_net(x)[0]
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class LinearAudioContinuation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_codebooks = config["n_codebooks"]
        self.vocab_size = config["vocab_size"]
        self.d_model = config.get("d_model", 512)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 6)
        self.dropout = config.get("dropout", 0.1)

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.max_seq_len = config.get("max_seq_len", 1000)
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, self.max_seq_len, self.d_model)
        )
        self.emb_dropout = nn.Dropout(self.dropout)

        self.blocks = nn.ModuleList(
            [
                LinearDecoderBlock(self.d_model, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = RMSNorm(self.d_model, eps=1e-6)
        self.output_head = nn.Linear(self.d_model, self.vocab_size)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # Handle 3D input: accept both [B, T, K] and canonical [B, K, T]
        if idx.dim() == 3:
            B = idx.shape[0]
            # If input is canonical [B, K, T], transpose to [B, T, K]
            if idx.shape[1] == self.n_codebooks:
                idx = idx.transpose(1, 2)
            # Flatten codebooks into sequence dimension: [B, T*K]
            idx = idx.reshape(B, -1)
            # Handle targets similarly
            if targets is not None and targets.dim() == 3:
                if targets.shape[1] == self.n_codebooks:
                    targets = targets.transpose(1, 2)
                targets = targets.reshape(B, -1)

        B, seq_len = idx.shape
        x = self.token_embedding(idx)

        if seq_len > self.max_seq_len:
            pos_emb = self.pos_embeddings[:, :seq_len, :]
        else:
            pos_emb = self.pos_embeddings[:, :seq_len, :]
        x = x + pos_emb
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            # Use reshape instead of view to handle non-contiguous tensors
            loss = self.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )
        return loss, logits

    def get_training_loss(self, past_tokens: torch.Tensor, future_tokens: torch.Tensor):
        """
        Accepts canonical [B, K, T] (codebooks, time) or [B, T, K] inputs.
        Concatenates along time dim and delegates shape handling to forward().
        """
        if past_tokens.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {past_tokens.dim()}D")

        past_tokens = past_tokens.long()
        future_tokens = future_tokens.long()

        # Concatenate along time dimension (dim=2 for [B,K,T], dim=1 for [B,T,K])
        # forward() will normalize shape internally
        all_tokens = torch.cat(
            [past_tokens, future_tokens], dim=-1
        )  # Auto-detect last dim as time

        # Create shifted inputs/targets along time dimension
        inputs_3d = (
            all_tokens[:, :, :-1]
            if all_tokens.shape[1] == self.n_codebooks
            else all_tokens[:, :-1, :]
        )
        targets_3d = (
            all_tokens[:, :, 1:]
            if all_tokens.shape[1] == self.n_codebooks
            else all_tokens[:, 1:, :]
        )

        loss, _ = self.forward(inputs_3d, targets=targets_3d)
        return loss

    @torch.no_grad()
    def predict(
        self,
        prompt_tokens: torch.Tensor,
        steps: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """
        prompt_tokens: Canonical [B, K, T] (codebooks, time) or [B, T, K]
        steps: number of generation steps (each step generates K tokens for all codebooks)
        temperature: softmax temperature scaling
        top_k: keep only the top_k highest logits, set rest to -inf
        top_p: nucleus sampling – keep the smallest set of tokens whose cumulative probability >= top_p
        """
        if prompt_tokens.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {prompt_tokens.dim()}D")

        # Normalize to [B, T, K] for flattening
        if prompt_tokens.shape[1] == self.n_codebooks:  # Canonical [B, K, T]
            prompt_tokens = prompt_tokens.transpose(1, 2)  # [B, T_prompt, K]

        B, T_prompt, K = prompt_tokens.shape
        generated_flat = prompt_tokens.reshape(B, -1)  # (B, T_prompt*K)

        total_flat_steps = steps * K
        for _ in range(total_flat_steps):
            _, logits = self.forward(generated_flat)
            logits_last = logits[:, -1, :] / temperature

            # top‑k filtering
            if top_k is not None:
                k = min(top_k, logits_last.size(-1))
                top_k_vals = torch.topk(logits_last, k)
                mask = logits_last < top_k_vals.values[:, -1:]
                logits_last[mask] = float("-inf")

            # nucleus (top‑p) filtering
            if top_p is not None:
                # compute probabilities after temperature and top‑k
                probs = F.softmax(logits_last, dim=-1)
                # sort probabilities descending
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum = sorted_probs.cumsum(dim=-1)
                # keep tokens where the cumulative mass before adding the token is < top_p
                # this guarantees we include the token that pushes the cumulative sum over top_p
                sorted_mask = cumsum - sorted_probs < top_p
                # scatter mask back to original vocabulary order
                mask = torch.zeros_like(logits_last, dtype=torch.bool)
                mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
                logits_last[~mask] = float("-inf")

            # sample from the filtered distribution
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_flat = torch.cat([generated_flat, next_token], dim=1)

        # Reshape back to (B, steps, K)
        generated = generated_flat[:, -total_flat_steps:].reshape(B, steps, K)
        return generated
