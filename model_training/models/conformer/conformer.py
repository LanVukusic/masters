import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Optional
from torch import Tensor

from model_training.models.conformer.feed_forward import FeedForwardModule
from model_training.models.conformer.attention import MultiHeadedSelfAttentionModule
from model_training.models.conformer.convolution import ConformerConvModule
from model_training.models.conformer.modules import ResidualConnectionModule


NUM_CONFORMER_LAYERS = 4
CONV_KERNEL_SIZE = 16
FEED_FORWARD_EXPANSION = 4


def create_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 256,
        num_attention_heads: int = 4,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 16,
        half_step_residual: bool = True,
    ):
        super().__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.sequential(inputs)


class AudioContinuationConformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_codebooks = config["n_codebooks"]
        self.vocab_size = config["vocab_size"]
        self.past_len = config.get("past_len", 200)
        self.future_len = config.get("future_len", 200)

        d_model = config.get("d_model", 256)
        num_layers = config.get("num_layers", NUM_CONFORMER_LAYERS)
        num_heads = config.get("n_heads", 4)
        dropout = config.get("dropout", 0.1)
        conv_kernel_size = config.get("conv_kernel_size", CONV_KERNEL_SIZE)
        if conv_kernel_size % 2 == 0:
            conv_kernel_size += 1

        self.codebook_embeddings = nn.ModuleList(
            [nn.Embedding(self.vocab_size, d_model) for _ in range(self.n_codebooks)]
        )

        max_len = self.past_len + self.future_len
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        self.conformer_layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=d_model,
                    num_attention_heads=num_heads,
                    feed_forward_expansion_factor=FEED_FORWARD_EXPANSION,
                    feed_forward_dropout_p=dropout,
                    attention_dropout_p=dropout,
                    conv_dropout_p=dropout,
                    conv_kernel_size=conv_kernel_size,
                    half_step_residual=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_heads = nn.ModuleList(
            [nn.Linear(d_model, self.vocab_size) for _ in range(self.n_codebooks)]
        )

        self.d_model = d_model
        self.num_layers = num_layers

        max_len = self.past_len + self.future_len
        self._causal_mask = None
        self._causal_mask_len = 0
        self._loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, past_tokens, future_tokens=None):
        """
        past_tokens: (Batch, T_past, N_Codebooks)
        future_tokens: (Batch, T_future, N_Codebooks) - optional, for teacher forcing
        """
        B, T_past, C = past_tokens.shape

        if future_tokens is not None:
            T_future = future_tokens.shape[1]
            combined = torch.cat([past_tokens, future_tokens], dim=1)
        else:
            combined = past_tokens
            T_future = 0

        T_total = combined.shape[1]

        cb_embs = []
        for cb_idx in range(C):
            cb_emb = self.codebook_embeddings[cb_idx](combined[:, :, cb_idx])
            cb_embs.append(cb_emb)

        x = sum(cb_embs) * math.sqrt(self.d_model)

        x = x + self.pos_embedding[:, :T_total, :]

        if T_total > 1:
            if self._causal_mask is None or self._causal_mask_len != T_total:
                self._causal_mask = create_causal_mask(T_total, x.device)
                self._causal_mask_len = T_total
            causal_mask = self._causal_mask
        else:
            causal_mask = None

        for layer in self.conformer_layers:
            x = layer(x, mask=causal_mask)

        if T_future > 0:
            x = x[:, T_past:, :]

        logits = []
        for head in self.output_heads:
            logits.append(head(x))

        return torch.stack(logits, dim=2)

    def get_training_loss(self, past_tokens, future_tokens):
        """
        past_tokens: (Batch, T_past, N_Codebooks)
        future_tokens: (Batch, T_future, N_Codebooks)
        """
        logits = self.forward(past_tokens, future_tokens)

        total_loss = 0
        for cb_idx in range(self.n_codebooks):
            reshaped_logits = logits[:, :, cb_idx, :].reshape(-1, self.vocab_size)
            t = future_tokens[:, :, cb_idx].reshape(-1)

            loss = self._loss_fn(reshaped_logits, t)
            weight = 0.7**cb_idx
            total_loss += weight * loss

        return total_loss

    def get_progressive_teacher_forcing_loss(
        self,
        past_tokens,
        future_tokens,
        teacher_forcing_ratio: float = 1.0,
    ):
        """
        Progressive teacher forcing: gradually transition from teacher forcing to free generation.

        Args:
            past_tokens: (Batch, T_past, N_Codebooks)
            future_tokens: (Batch, T_future, N_Codebooks)
            teacher_forcing_ratio: 1.0 = full teacher forcing, 0.0 = full autoregressive

        Memory-efficient version: compute loss on full sequence first, then do a second forward for sampling.
        """
        B, T_future, C = future_tokens.shape

        logits = self.forward(past_tokens, future_tokens)

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        total_loss = 0

        for cb_idx in range(C):
            l = logits[:, :, cb_idx, :].reshape(-1, self.vocab_size)
            t = future_tokens[:, :, cb_idx].reshape(-1)
            loss = loss_fn(l, t)
            weight = 0.7**cb_idx
            total_loss += weight * loss

        if teacher_forcing_ratio < 1.0 and teacher_forcing_ratio > 0.0:
            num_free_samples = max(1, int(T_future * (1 - teacher_forcing_ratio)))
            sample_indices = random.sample(range(T_future), num_free_samples)

            generated = past_tokens.clone()
            free_loss = 0

            for t in range(T_future):
                with torch.no_grad():
                    logits = self.forward(generated)
                    last_logits = logits[:, -1, :, :]

                    if t in sample_indices:
                        for cb_idx in range(C):
                            l = last_logits[:, cb_idx, :].detach()
                            t_idx = future_tokens[:, t, cb_idx]
                            loss = loss_fn(l, t_idx)
                            weight = 0.7**cb_idx
                            free_loss = free_loss + weight * loss

                        next_tokens = []
                        for cb_idx in range(C):
                            probs = torch.softmax(last_logits[:, cb_idx, :], dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            next_tokens.append(next_token)
                        next_frame = torch.stack(next_tokens, dim=-1)
                    else:
                        next_frame = future_tokens[:, t, :].unsqueeze(1)

                    generated = torch.cat([generated, next_frame], dim=1)

            free_loss = free_loss / (num_free_samples * C)
            total_loss = (
                teacher_forcing_ratio * total_loss
                + (1 - teacher_forcing_ratio) * free_loss
            )

        return total_loss

    def predict(
        self,
        prompt_tokens,
        temperature=1.0,
        top_k=None,
        top_p=0.95,
        repetition_penalty=1.2,
        predict_by_one=False,
        true_future_tokens: Optional[torch.Tensor] = None,
    ):
        """
        Autoregressive generation.
        prompt_tokens: (Batch, T_prompt, N_Codebooks)
        predict_by_one: if True, model will predict every frame on true frames, not relying on self generated ones
        """

        if predict_by_one and not (
            true_future_tokens is not None and true_future_tokens.numel() > 0
        ):
            raise Exception("If predict by one is set, true future tokens are needed")

        max_new_tokens = self.future_len
        self.eval()

        batch_size, time, codebooks = prompt_tokens.shape

        generated = prompt_tokens.clone()
        out = torch.empty(0, dtype=torch.long, device=prompt_tokens.device)
        t = 0

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated)

                last_logits = logits[:, -1, :, :]

                next_tokens = []
                for cb_idx in range(self.n_codebooks):
                    cb_logits = last_logits[:, cb_idx, :].clone()

                    if repetition_penalty != 1.0:
                        past_tokens = generated[:, :, cb_idx]
                        for b in range(batch_size):
                            unique_tokens = torch.unique(past_tokens[b])
                            cb_logits[b, unique_tokens] /= repetition_penalty

                    cb_logits = cb_logits / temperature

                    if top_k is not None and top_k > 0:
                        top_k_vals = torch.topk(
                            cb_logits, min(top_k, cb_logits.size(-1))
                        )
                        cb_logits[cb_logits < top_k_vals.values[:, -1:]] = float("-inf")

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            cb_logits, descending=True
                        )
                        probs_sorted = nn.functional.softmax(sorted_logits, dim=-1)
                        cumsum_probs = torch.cumsum(probs_sorted, dim=-1)

                        sorted_indices_to_remove = cumsum_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = False

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        cb_logits[indices_to_remove] = float("-inf")

                    probs = torch.softmax(cb_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_tokens.append(next_token)

                next_frame = torch.stack(next_tokens, dim=-1)
                out = torch.cat([out, next_frame], dim=1)
                if predict_by_one:
                    assert true_future_tokens is not None
                    next_true_frame = true_future_tokens[:, t, :].unsqueeze(1)
                    generated = torch.cat([generated, next_true_frame], dim=1)
                else:
                    generated = torch.cat([generated, next_frame], dim=1)

                t += 1

        return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from torchinfo import summary
    from model_training.models.conformer.conformer import AudioContinuationConformer

    config = {
        "n_codebooks": 4,
        "vocab_size": 1024,
        "past_len": 200,
        "future_len": 200,
        "d_model": 256,
        "n_heads": 4,
        "dropout": 0.1,
        "num_layers": 4,
    }

    model = AudioContinuationConformer(config)
    x = torch.randint(0, 1024, (16, 200, 4))
    summary(model, input_data=x)
