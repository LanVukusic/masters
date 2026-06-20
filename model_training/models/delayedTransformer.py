import torch
import torch.nn as nn

from models.causal_transformer import CausalTransformer


class AudioContinuationTransformerDelay(nn.Module):
    """
    Causal transformer with delay‑pattern flattening for multi‑codebook audio tokens.
    Input/output shapes remain [B, T, K] for convenience; the model internally
    flattens to [B, T*K] and uses a single embedding + head.
    """

    def __init__(self, config):
        super().__init__()
        self.n_codebooks = config["n_codebooks"]
        self.vocab_size = config["vocab_size"]
        self.past_len = config.get("past_len", 200)
        self.future_len = config.get("future_len", 200)

        d_model = config.get("d_model", 768)
        n_heads = config.get("n_heads", 12)
        d_ff = config.get("d_ff", 2048)
        n_layers = config.get("n_layers", 12)
        dropout = config.get("dropout", 0.1)

        # Maximum flattened sequence length
        self.max_len = (self.past_len + self.future_len) * self.n_codebooks
        self.d_model = d_model

        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_len, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        self.transformer = CausalTransformer(d_model, n_heads, d_ff, n_layers, dropout)
        self.final_norm = nn.LayerNorm(d_model)

        self.output_head = nn.Linear(d_model, self.vocab_size)

        self._loss_fn = nn.CrossEntropyLoss()

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_head.bias)
        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _flatten(tokens: torch.Tensor) -> torch.Tensor:
        B, T, K = tokens.shape
        return tokens.permute(0, 2, 1).reshape(B, T * K)

    @staticmethod
    def _unflatten(flat: torch.Tensor, K: int) -> torch.Tensor:
        B, L = flat.shape
        T = L // K
        return flat.reshape(B, K, T).permute(0, 2, 1)

    @staticmethod
    def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
        B, T, K, V = logits.shape
        return logits.permute(0, 2, 1, 3).reshape(B, T * K, V)

    @staticmethod
    def _unflatten_logits(flat_logits: torch.Tensor, K: int) -> torch.Tensor:
        B, L, V = flat_logits.shape
        T = L // K
        return flat_logits.reshape(B, K, T, V).permute(0, 2, 1, 3)

    def _embed_flat(
        self, flat_tokens: torch.Tensor, pos_offset: int = 0
    ) -> torch.Tensor:
        L = flat_tokens.shape[1]
        total_pos = L + pos_offset
        if total_pos > self.max_len:
            raise ValueError(
                f"Flattened length {L} at offset {pos_offset} exceeds max {self.max_len}"
            )
        x = self.token_embedding(flat_tokens)
        x = x + self.pos_embedding[:, pos_offset:total_pos, :]
        return self.embed_dropout(x)

    def _run_backbone_flat(
        self,
        flat_tokens: torch.Tensor,
        kv_caches: list[dict] | None = None,
        pos_offset: int = 0,
    ):
        x = self._embed_flat(flat_tokens, pos_offset=pos_offset)
        x, new_caches = self.transformer(x, kv_caches=kv_caches)
        x = self.final_norm(x)
        return x, new_caches

    def forward(self, past_tokens, future_tokens=None):
        if future_tokens is None:
            flat_past = self._flatten(past_tokens)
            hidden, _ = self._run_backbone_flat(flat_past)
            logits = self.output_head(hidden)  # [B, T_past*K, V]
            return self._unflatten_logits(logits, self.n_codebooks)

        T_past = past_tokens.shape[1]
        T_future = future_tokens.shape[1]
        full = torch.cat([past_tokens, future_tokens], dim=1)  # [B, T_past+T_future, K]
        flat_full = self._flatten(full)  # [B, (T_past+T_future)*K]

        # Drop last token -> input, predict next token at each position
        hidden, _ = self._run_backbone_flat(
            flat_full[:, :-1]
        )  # [B, (T_past+T_future)*K - 1, d_model]
        logits = self.output_head(hidden)  # [B, (T_past+T_future)*K - 1, V]

        # The first future token is at flat position T_past*K.
        # The logit that predicts it is at position T_past*K - 1.
        start_idx = T_past * self.n_codebooks - 1
        end_idx = start_idx + T_future * self.n_codebooks  # = (T_past+T_future)*K - 1
        future_logits = logits[:, start_idx:end_idx, :]  # [B, T_future*K, V]
        return self._unflatten_logits(future_logits, self.n_codebooks)

    def get_training_loss(self, past_tokens, future_tokens, return_metrics=False):
        logits_3d = self.forward(past_tokens, future_tokens)  # [B, T_future, K, V]
        flat_logits = self._flatten_logits(logits_3d)  # [B, T_future*K, V]
        flat_targets = self._flatten(future_tokens)  # [B, T_future*K]

        total_loss = self._loss_fn(
            flat_logits.reshape(-1, self.vocab_size),
            flat_targets.reshape(-1),
        )

        if return_metrics:
            with torch.no_grad():
                top_k = min(5, self.vocab_size)
                topk_idx = flat_logits.topk(
                    top_k, dim=-1
                ).indices  # [B, T_future*K, top_k]
                targets = flat_targets  # [B, T_future*K]
                correct = (topk_idx[..., 0] == targets).float().mean()
                top5 = (topk_idx == targets.unsqueeze(-1)).any(-1).float().mean()

                per_cb_loss = torch.full(
                    (self.n_codebooks,), total_loss.item(), device=flat_logits.device
                )
                per_cb_top1 = torch.full(
                    (self.n_codebooks,), correct.item(), device=flat_logits.device
                )
                per_cb_top5 = torch.full(
                    (self.n_codebooks,), top5.item(), device=flat_logits.device
                )

            return (
                total_loss,
                {
                    "Loss": per_cb_loss,
                    "Top1": per_cb_top1,
                    "Top5": per_cb_top5,
                },
                logits_3d,
            )
        return total_loss, logits_3d

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
        Autoregressive generation with delay pattern.
        Returns: [B, T_out, K]
        """
        if predict_by_one and (
            true_future_tokens is None or true_future_tokens.numel() == 0
        ):
            raise ValueError("predict_by_one=True requires true_future_tokens.")

        was_training = self.training
        self.eval()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # --- predict_by_one: single forward pass (teacher forcing) ---
            if predict_by_one:
                logits = self.forward(
                    prompt_tokens, true_future_tokens
                )  # [B, T_future, K, V]
                if temperature == 0.0:
                    preds = logits.argmax(dim=-1)
                else:
                    logits = logits / max(temperature, 1e-5)
                    if top_k is not None and top_k > 0:
                        kth = torch.topk(
                            logits, min(top_k, self.vocab_size), dim=-1
                        ).values[..., -1:]
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    if top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(
                            logits, descending=True, dim=-1
                        )
                        cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        remove = cum > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        drop_mask = torch.zeros_like(remove).scatter(
                            -1, sorted_idx, remove
                        )
                        logits = logits.masked_fill(drop_mask, float("-inf"))
                    probs = torch.softmax(logits, dim=-1)
                    flat_probs = probs.reshape(-1, self.vocab_size)
                    flat_samples = torch.multinomial(flat_probs, num_samples=1)
                    preds = flat_samples.view(probs.shape[:-1])
                self.train(was_training)
                return preds

            # --- Free-running AR generation with KV caching ---
            max_context = self.max_len - 1
            flat_prompt = self._flatten(prompt_tokens)
            if flat_prompt.shape[1] > max_context:
                flat_prompt = flat_prompt[:, -max_context:]
            prompt_len = flat_prompt.shape[1]

            max_new_tokens = self.future_len * self.n_codebooks
            if true_future_tokens is not None:
                max_new_tokens = min(
                    max_new_tokens, true_future_tokens.shape[1] * self.n_codebooks
                )
            max_new_tokens = min(max_new_tokens, self.max_len - prompt_len)
            if max_new_tokens <= 0:
                raise ValueError("No room to generate.")

            # Tracks all generated tokens for repetition penalty
            generated = flat_prompt.clone()
            out_tokens = []

            # Step 1: full prompt forward, populate KV caches
            hidden, caches = self._run_backbone_flat(flat_prompt, pos_offset=0)
            logits_last = self.output_head(hidden[:, -1:, :])[:, 0, :]

            logits = logits_last.clone()
            if repetition_penalty != 1.0:
                seen = torch.unique(generated)
                logits[:, seen] /= repetition_penalty
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                kth = torch.topk(logits, min(top_k, self.vocab_size)).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                drop_mask = torch.zeros_like(remove).scatter(1, sorted_idx, remove)
                logits = logits.masked_fill(drop_mask, float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            out_tokens.append(next_token)
            generated = torch.cat([generated, next_token], dim=1)

            # Steps 2..N: incremental with KV caches (1 new token per step)
            for _ in range(1, max_new_tokens):
                hidden, caches = self._run_backbone_flat(
                    next_token, kv_caches=caches, pos_offset=generated.shape[1] - 1
                )
                logits_last = self.output_head(hidden[:, -1:, :])[:, 0, :]

                logits = logits_last.clone()
                if repetition_penalty != 1.0:
                    seen = torch.unique(generated)
                    logits[:, seen] /= repetition_penalty
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    kth = torch.topk(logits, min(top_k, self.vocab_size)).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    drop_mask = torch.zeros_like(remove).scatter(1, sorted_idx, remove)
                    logits = logits.masked_fill(drop_mask, float("-inf"))

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                out_tokens.append(next_token)
                generated = torch.cat([generated, next_token], dim=1)

            del hidden, caches

        self.train(was_training)

        out_flat = torch.cat(out_tokens, dim=1)  # [B, max_new_tokens]
        T_out = out_flat.shape[1] // self.n_codebooks
        out_flat = out_flat[:, : T_out * self.n_codebooks]
        return self._unflatten(out_flat, self.n_codebooks)


if __name__ == "__main__":
    from torchinfo import summary

    config = {
        "n_codebooks": 4,
        "vocab_size": 1024,
        "past_len": 1200,
        "future_len": 225,
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "d_ff": 512,
        "dropout": 0.1,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioContinuationTransformerDelay(config).to(device)
    print(f"Device: {device}")
    print(f"max_len (flattened): {model.max_len}")
    print(
        f"  = (past_len={config['past_len']} + future_len={config['future_len']}) × n_codebooks={config['n_codebooks']}"
    )
    print(
        f"  = {config['past_len'] + config['future_len']} time steps × {config['n_codebooks']} codebooks"
    )
    print()

    B, Tp, Tf, K = 2, config["past_len"], config["future_len"], config["n_codebooks"]
    past = torch.randint(0, config["vocab_size"], (B, Tp, K), device=device)
    future = torch.randint(0, config["vocab_size"], (B, Tf, K), device=device)

    summary(
        model,
        input_data=[past, future],
        depth=4,
        col_names=["input_size", "output_size", "num_params", "trainable"],
    )

    print("\n--- Memory estimate ---")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Parameter memory (fp32): {sum(p.numel() for p in model.parameters()) * 4 / 1e9:.2f} GB"
    )
    print(
        f"  Position embedding size: {model.pos_embedding.numel() * model.pos_embedding.element_size() / 1e6:.1f} MB"
    )
    print(
        f"  Peak CUDA memory after init: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        if torch.cuda.is_available()
        else ""
    )
