import torch
import torch.nn as nn


class AudioContinuationTransformer(nn.Module):
    """
    Small causal transformer for multi-codebook audio token continuation.

    Shape contract (see AGENTS.md):
      past_tokens:   [B, T_past,   K]
      future_tokens: [B, T_future, K]

    forward returns:
      [B, T_future, K, V]  when future_tokens is given (logits aligned to targets)
      [B, T_in,    K, V]  when future_tokens is None  (logits at every input position)
    """

    def __init__(self, config):
        super().__init__()
        self.n_codebooks = config["n_codebooks"]
        self.vocab_size = config["vocab_size"]
        self.past_len = config.get("past_len", 200)
        self.future_len = config.get("future_len", 200)

        d_model = config.get("d_model", 256)
        n_heads = config.get("n_heads", 4)
        d_ff = config.get("d_ff", 4 * d_model)
        n_layers = config.get("n_layers", 4)
        dropout = config.get("dropout", 0.1)

        self.max_len = self.past_len + self.future_len
        self.d_model = d_model

        self.codebook_embeddings = nn.ModuleList(
            [nn.Embedding(self.vocab_size, d_model) for _ in range(self.n_codebooks)]
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_len, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        self.output_heads = nn.ModuleList(
            [nn.Linear(d_model, self.vocab_size) for _ in range(self.n_codebooks)]
        )

        self._loss_fn = nn.CrossEntropyLoss()
        self._init_weights()

    def _init_weights(self):
        """GPT-2 style small init so token and positional signals are balanced
        in magnitude. nn.Embedding default is N(0, 1), which makes the summed
        token embedding ~100x larger than a randn(..) * 0.02 pos embedding —
        positional information ends up effectively invisible to the model.
        """
        for emb in self.codebook_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        for head in self.output_heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(head.bias)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, K] -> [B, T, d_model]
        T = tokens.shape[1]
        if T > self.max_len:
            raise ValueError(
                f"Sequence length {T} exceeds model max_len {self.max_len}."
            )
        x = sum(
            self.codebook_embeddings[k](tokens[:, :, k])
            for k in range(self.n_codebooks)
        )
        x = x + self.pos_embedding[:, :T, :]
        return self.embed_dropout(x)

    def _run_backbone(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self._embed(tokens)
        T = x.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device, dtype=x.dtype
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.final_norm(x)  # [B, T, d_model]

    def _apply_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, d_model] -> [B, T, K, V]
        return torch.stack([head(x) for head in self.output_heads], dim=2)

    def forward(self, past_tokens, future_tokens=None):
        if future_tokens is None:
            x = self._run_backbone(past_tokens)
            return self._apply_heads(x)

        T_past = past_tokens.shape[1]
        T_future = future_tokens.shape[1]

        # Standard CLM shift: feed [past, future[:-1]] and read positions
        # [T_past - 1, T_past - 1 + T_future). The logit at position p is
        # conditioned on tokens 0..p and predicts the token at position p+1.
        inputs = torch.cat([past_tokens, future_tokens[:, :-1, :]], dim=1)
        x = self._run_backbone(inputs)
        x = x[:, T_past - 1 : T_past - 1 + T_future, :]
        return self._apply_heads(x)

    def get_training_loss(
        self,
        past_tokens,
        future_tokens,
        return_metrics: bool = False,
    ):
        """
        Returns the mean CE loss across codebooks.
        If return_metrics=True, also returns a dict with per-codebook tensors:
          {"Loss": [K], "Top1": [K], "Top5": [K]}.
        """
        logits = self.forward(past_tokens, future_tokens)  # [B, T_future, K, V]
        total = 0.0
        per_cb_loss = [] if return_metrics else None
        per_cb_top1 = [] if return_metrics else None
        per_cb_top5 = [] if return_metrics else None
        top_k = min(5, self.vocab_size)

        for k in range(self.n_codebooks):
            cb_logits = logits[:, :, k, :].reshape(-1, self.vocab_size)
            cb_targets = future_tokens[:, :, k].reshape(-1)
            loss_k = self._loss_fn(cb_logits, cb_targets)
            total = total + loss_k

            if return_metrics:
                with torch.no_grad():
                    topk_idx = cb_logits.topk(top_k, dim=-1).indices  # [N, top_k]
                    targets_expanded = cb_targets.unsqueeze(-1)
                    per_cb_loss.append(loss_k.detach())
                    per_cb_top1.append((topk_idx[:, 0] == cb_targets).float().mean())
                    per_cb_top5.append(
                        (topk_idx == targets_expanded).any(-1).float().mean()
                    )

        total = total / self.n_codebooks
        if return_metrics:
            return total, {
                "Loss": torch.stack(per_cb_loss),
                "Top1": torch.stack(per_cb_top1),
                "Top5": torch.stack(per_cb_top5),
            }
        return total

    @torch.no_grad()
    def predict(
        self,
        prompt_tokens,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        predict_by_one: bool = False,
        true_future_tokens: torch.Tensor | None = None,
    ):
        """
        Autoregressive generation.
          prompt_tokens: [B, T_prompt, K]
          returns:       [B, T_out,    K]
        If predict_by_one=True, ground-truth frames are fed back as context
        (next-token diagnostic) but sampled frames are still returned.
        """
        if predict_by_one and (
            true_future_tokens is None or true_future_tokens.numel() == 0
        ):
            raise ValueError("predict_by_one=True requires true_future_tokens.")

        was_training = self.training
        self.eval()

        max_context = self.max_len - 1
        if prompt_tokens.shape[1] > max_context:
            prompt_tokens = prompt_tokens[:, -max_context:, :].contiguous()
        prompt_len = prompt_tokens.shape[1]

        max_new = self.future_len
        if true_future_tokens is not None:
            max_new = min(max_new, true_future_tokens.shape[1])
        max_new = min(max_new, self.max_len - prompt_len)
        if max_new <= 0:
            raise ValueError("No room to generate: prompt fills the context window.")

        generated = prompt_tokens.clone()
        out_frames = []

        for t in range(max_new):
            hidden = self._run_backbone(generated)               # [B, T, d_model]
            last_logits = self._apply_heads(hidden[:, -1:, :])[:, 0]  # [B, K, V]

            next_frame = self._sample_frame(
                last_logits,
                generated,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )  # [B, 1, K]
            out_frames.append(next_frame)

            ctx_frame = (
                true_future_tokens[:, t : t + 1, :] if predict_by_one else next_frame
            )
            generated = torch.cat([generated, ctx_frame], dim=1)

        if was_training:
            self.train()

        return torch.cat(out_frames, dim=1)  # [B, T_out, K]

    def _sample_frame(
        self,
        logits: torch.Tensor,      # [B, K, V]
        generated: torch.Tensor,   # [B, T, K]
        temperature: float,
        top_k: int | None,
        top_p: float,
        repetition_penalty: float,
    ) -> torch.Tensor:
        B, K, V = logits.shape
        sampled = []
        for k in range(K):
            cb_logits = logits[:, k, :].clone()

            if repetition_penalty != 1.0:
                past = generated[:, :, k]
                for b in range(B):
                    seen = torch.unique(past[b])
                    cb_logits[b, seen] /= repetition_penalty

            if temperature != 1.0:
                cb_logits = cb_logits / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                kth = torch.topk(cb_logits, min(top_k, V)).values[:, -1:]
                cb_logits = cb_logits.masked_fill(cb_logits < kth, float("-inf"))

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(cb_logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                drop_mask = torch.zeros_like(remove).scatter(1, sorted_idx, remove)
                cb_logits = cb_logits.masked_fill(drop_mask, float("-inf"))

            probs = torch.softmax(cb_logits, dim=-1)
            sampled.append(torch.multinomial(probs, num_samples=1))  # [B, 1]

        return torch.stack(sampled, dim=-1)  # [B, 1, K]


if __name__ == "__main__":
    config = {
        "n_codebooks": 12,
        "vocab_size": 1024,
        "past_len": 150,
        "future_len": 150,
        "d_model": 256,
        "n_heads": 4,
        "d_ff": 1024,
        "n_layers": 4,
        "dropout": 0.1,
    }
    model = AudioContinuationTransformer(config)
    past = torch.randint(0, 1024, (2, 150, 12))
    future = torch.randint(0, 1024, (2, 150, 12))
    out = model(past, future)
    print("forward (with future):", out.shape)  # [2, 150, 12, 1024]
    loss = model.get_training_loss(past, future)
    print("loss:", loss.item())
    gen = model.predict(past, top_k=200, top_p=0.95)
    print("predict:", gen.shape)  # [2, 150, 12]
