import torch
import torch.nn as nn


class AudioContinuationTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_codebooks = config["n_codebooks"]
        self.vocab_size = config["vocab_size"]
        self.d_model = 512  # Much larger than 128

        # 1. EMBEDDINGS (Critical Fix)
        # Separate embedding table for each codebook
        self.codebook_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.vocab_size, self.d_model)
                for _ in range(self.n_codebooks)
            ]
        )

        # 2. TEMPORAL POSITION EMBEDDINGS
        self.pos_embedding = nn.Embedding(2048, self.d_model)  # Max seq len

        # 3. TRANSFORMER DECODER (Autoregressive Fix)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # 4. OUTPUT HEADS
        # Project back to vocab size for each codebook
        self.output_heads = nn.ModuleList(
            [nn.Linear(self.d_model, self.vocab_size) for _ in range(self.n_codebooks)]
        )

    def forward(self, tokens, mask=None):
        """
        tokens: (Batch, Seq_Len, N_Codebooks)
        """
        B, T, C = tokens.shape

        # 1. Embed each codebook separately
        codebook_embs = []
        for cb_idx in range(C):
            cb_tokens = tokens[:, :, cb_idx]
            cb_emb = self.codebook_embeddings[cb_idx](cb_tokens)
            codebook_embs.append(cb_emb)

        # Sum embeddings across codebooks (Standard MusicGen/AudioLM approach)
        # This allows information mixing between codebooks at the input level
        x = sum(codebook_embs)

        # 2. Add Positional Embeddings
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # 3. Transformer Pass
        # tgt_mask should be causal (triangular) to prevent looking ahead
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tokens.device
        )
        x = self.transformer_decoder(x, memory=x, tgt_mask=tgt_mask)

        # 4. Project to logits
        logits = []
        for head in self.output_heads:
            logits.append(head(x))

        # Stack: (Batch, Seq_Len, N_Codebooks, Vocab)
        return torch.stack(logits, dim=2)

    def get_training_loss(self, past_tokens, future_tokens):
        """
        past_tokens: Input sequence (Batch, T, C)
        future_tokens: Target sequence (Batch, T, C) - usually tokens shifted by 1
        """
        logits = self.forward(past_tokens)  # (B, T, C, V)
        # fts = future_tokens.shape
        # [A,B,C,D] => [B,C,D,E], where E is future
        future_tokens = torch.concat(
            (past_tokens[:, 1:, :], future_tokens[:, 0, :].unsqueeze(1)), dim=1
        )
        # print("ftokens", fts, past_tokens.shape, "-->", future_tokens.shape)

        total_loss = 0
        loss_fn = nn.CrossEntropyLoss()

        # Calculate loss per codebook
        for cb_idx in range(self.n_codebooks):
            # Reshape to (B*T, V) and (B*T)
            l = logits[:, :, cb_idx, :].reshape(-1, self.vocab_size)
            t = future_tokens[:, :, cb_idx].reshape(-1).long()

            # Optional: Fidelity decay weighting
            weight = 0.5**cb_idx
            total_loss += weight * loss_fn(l, t)

        return total_loss

        # --- NEW PREDICT METHOD ---

    def predict(self, prompt_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation loop.

        Args:
            prompt_tokens: (Batch, Prompt_Len, N_Codebooks) - The context audio
            max_new_tokens: int - How many future frames to generate
            temperature: float - Sampling temperature
            top_k: int - Top-K filtering

        Returns:
            generated_tokens: (Batch, max_new_tokens, N_Codebooks) - The continuation
        """

        max_new_tokens = 100
        self.eval()
        device = prompt_tokens.device
        batch_size = prompt_tokens.shape[0]

        # Start with the prompt
        # current_seq shape: (Batch, Current_Len, N_Codebooks)
        current_seq = prompt_tokens.clone()

        generated_frames = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                current_len = current_seq.shape[1]

                # 1. Forward pass on the entire sequence so far
                # We only need the logits for the LAST position
                logits = self.forward(current_seq)  # (B, L, C, V)

                # Extract logits for the last timestep
                last_logits = logits[:, -1, :, :]  # (B, C, V)

                # 2. Sample for each codebook independently
                next_tokens = []
                for cb_idx in range(self.n_codebooks):
                    cb_logits = last_logits[:, cb_idx, :] / temperature

                    # Top-K Filtering
                    if top_k is not None:
                        indices_to_remove = (
                            cb_logits < torch.topk(cb_logits, top_k)[0][..., -1, None]
                        )
                        cb_logits[indices_to_remove] = float("-inf")

                    # Convert to probabilities
                    probs = nn.functional.softmax(cb_logits, dim=-1)

                    # Sample
                    # multinomial expects (N, C) where C is prob dist size
                    # probs is (Batch, Vocab)
                    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                    next_tokens.append(next_token)

                # Stack codebooks back together
                # next_frame shape: (Batch, 1, N_Codebooks)
                next_frame = torch.stack(next_tokens, dim=-1)

                # 3. Append to sequence
                current_seq = torch.cat([current_seq, next_frame], dim=1)

                # 4. Store generated frame (excluding prompt)
                generated_frames.append(next_frame)

                # Optional: Break if EOS token encountered (if you have one)
                # if torch.all(next_frame == eos_token_id): break

        # Concatenate all generated frames
        # Result shape: (Batch, max_new_tokens, N_Codebooks)
        return torch.cat(generated_frames, dim=1)
