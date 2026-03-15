import torch
import torch.nn as nn
import math


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
            l = logits[:, :, cb_idx, :].reshape(-1, self.vocab_size)
            t = future_tokens[:, :, cb_idx].reshape(-1)

            loss = loss_fn(l, t)
            weight = 0.7**cb_idx
            total_loss += weight * loss

        return total_loss

    def predict(self, prompt_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation.
        prompt_tokens: (Batch, T_prompt, N_Codebooks)
        """
        max_new_tokens = self.future_len
        self.eval()

        generated = prompt_tokens.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(generated)

                last_logits = logits[:, -1, :, :]

                next_tokens = []
                for cb_idx in range(self.n_codebooks):
                    cb_logits = last_logits[:, cb_idx, :] / temperature

                    if top_k is not None:
                        top_k_vals = torch.topk(
                            cb_logits, min(top_k, cb_logits.size(-1))
                        )
                        cb_logits[cb_logits < top_k_vals.values[:, -1:]] = float("-inf")

                    probs = torch.softmax(cb_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_tokens.append(next_token)

                next_frame = torch.stack(next_tokens, dim=-1)
                generated = torch.cat([generated, next_frame], dim=1)

        return generated[:, -max_new_tokens:, :]
