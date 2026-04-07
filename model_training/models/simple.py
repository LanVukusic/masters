import torch
import torch.nn as nn
import math
import random


class AudioContinuationTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_codebooks = config["n_codebooks"]
        self.vocab_size = config["vocab_size"]
        self.past_len = config.get("past_len", 200)
        self.future_len = config.get("future_len", 200)

        d_model = config.get("d_model", 256)

        self.codebook_embeddings = nn.ModuleList(
            [nn.Embedding(self.vocab_size, d_model) for _ in range(self.n_codebooks)]
        )

        max_len = self.past_len + self.future_len
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.get("n_heads", 4),
            dim_feedforward=config.get("d_ff", 512),
            dropout=config.get("dropout", 0.1),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.output_heads = nn.ModuleList(
            [nn.Linear(d_model, self.vocab_size) for _ in range(self.n_codebooks)]
        )

        self.d_model = d_model

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

        x = self.transformer(x)

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

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        total_loss = 0
        for cb_idx in range(self.n_codebooks):
            reshaped_logits = logits[:, :, cb_idx, :].reshape(-1, self.vocab_size)
            t = future_tokens[:, :, cb_idx].reshape(-1)

            loss = loss_fn(reshaped_logits, t)
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

        # Full teacher forcing: concatenate past + future, compute loss on all future positions
        combined = torch.cat([past_tokens, future_tokens], dim=1)
        logits = self.forward(past_tokens, future_tokens)  # [B, T_future, C, V]

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        total_loss = 0

        for cb_idx in range(C):
            l = logits[:, :, cb_idx, :].reshape(-1, self.vocab_size)
            t = future_tokens[:, :, cb_idx].reshape(-1)
            loss = loss_fn(l, t)
            weight = 0.7**cb_idx
            total_loss += weight * loss

        # If teacher_forcing_ratio < 1.0, we need to add an additional loss for free-running generation
        # Sample some positions and compute loss with model's own predictions
        if teacher_forcing_ratio < 1.0 and teacher_forcing_ratio > 0.0:
            num_free_samples = max(1, int(T_future * (1 - teacher_forcing_ratio)))
            sample_indices = random.sample(range(T_future), num_free_samples)

            generated = past_tokens.clone()
            free_loss = 0

            for t in range(T_future):
                # Use no_grad to prevent memory buildup during autoregressive generation
                with torch.no_grad():
                    logits = self.forward(generated)
                    last_logits = logits[:, -1, :, :]

                    if t in sample_indices:
                        # Compute loss using ground truth (detach to not build graph for this part)
                        for cb_idx in range(C):
                            l = last_logits[:, cb_idx, :].detach()
                            t_idx = future_tokens[:, t, cb_idx]
                            loss = loss_fn(l, t_idx)
                            weight = 0.7**cb_idx
                            free_loss = free_loss + weight * loss

                        # Use model's own prediction for next input
                        next_tokens = []
                        for cb_idx in range(C):
                            probs = torch.softmax(last_logits[:, cb_idx, :], dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            next_tokens.append(next_token)
                        next_frame = torch.stack(next_tokens, dim=-1)
                    else:
                        # Use ground truth
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
        true_future_tokens: torch.Tensor = None,
    ):
        """
        Autoregressive generation.
        prompt_tokens: (Batch, T_prompt, N_Codebooks)
        predict_by_one: if True, model will predict every frame on true frames, not relying on self generated ones
        """

        if predict_by_one and not true_future_tokens.numel() > 0:
            raise Exception("If predict by one is set, true future tokens are needed")

        max_new_tokens = self.future_len
        self.eval()

        batch_size, time, codebooks = prompt_tokens.shape

        generated = prompt_tokens.clone()
        out = torch.Tensor().to(torch.long).to(prompt_tokens.device)
        t = 0

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated)

                last_logits = logits[:, -1, :, :]

                next_tokens = []
                for cb_idx in range(self.n_codebooks):
                    cb_logits = last_logits[:, cb_idx, :].clone()

                    # --- Vectorized Repetition Penalty ---
                    if repetition_penalty != 1.0:
                        # Get all past tokens for this codebook: (Batch, Current_Sequence_Length)
                        past_tokens = generated[:, :, cb_idx]
                        # We want to divide logits[token_id] by penalty.
                        # Equivalent to: logits -= log(penalty) if working in log-probs,
                        # but here we do division on logits directly as per your logic.
                        # To vectorize: gather the penalty values for the past tokens and apply.
                        # Simpler approach for standard penalty:
                        for b in range(batch_size):
                            unique_tokens = torch.unique(past_tokens[b])
                            cb_logits[b, unique_tokens] /= repetition_penalty

                        cb_logits = cb_logits / temperature

                    # Top-k filtering
                    if top_k is not None and top_k > 0:
                        top_k_vals = torch.topk(
                            cb_logits, min(top_k, cb_logits.size(-1))
                        )
                        cb_logits[cb_logits < top_k_vals.values[:, -1:]] = float("-inf")

                    # Nucleus (top-p) sampling
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

                        # Scatter back to original indices
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        cb_logits[indices_to_remove] = float("-inf")

                    # sample
                    probs = torch.softmax(cb_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_tokens.append(next_token)

                next_frame = torch.stack(next_tokens, dim=-1)
                out = torch.cat([out, next_frame], dim=1)
                if predict_by_one:
                    next_true_frame = true_future_tokens[:, t, :].unsqueeze(1)
                    generated = torch.cat([generated, next_true_frame], dim=1)
                else:
                    generated = torch.cat([generated, next_frame], dim=1)

                t += 1

        return out
