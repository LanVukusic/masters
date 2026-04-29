import torch
import torch.nn.functional as F

class SamplingProcessor:
    def __init__(self, temperature=1.0, top_k=None, top_p=0.95, repetition_penalty=1.2):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

    def __call__(self, logits, past_tokens=None):
        """Modify logits in-place or return filtered logits."""
        # 1. Repetition penalty
        if self.repetition_penalty != 1.0 and past_tokens is not None:
            for b in range(logits.size(0)):
                unique_tokens = torch.unique(past_tokens[b])
                logits[b, unique_tokens] /= self.repetition_penalty

        # 2. Temperature
        logits = logits / self.temperature

        # 3. Top-k
        if self.top_k is not None and self.top_k > 0:
            topk = torch.topk(logits, min(self.top_k, logits.size(-1)))
            threshold = topk.values[:, -1:]          # (B,1)
            logits[logits < threshold] = float('-inf')

        # 4. Top-p (nucleus)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs_sorted = F.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(probs_sorted, dim=-1)

            sorted_mask = cumsum_probs > self.top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False

            indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits[indices_to_remove] = float('-inf')

        return logits