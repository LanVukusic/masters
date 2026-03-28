import sys
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "model_training")
)

from simpleModel.simple_v2 import AudioContinuationTransformer
from model_training.tokenizer.dac_audio_tokenizer import DACAudioTokenizer
from config import EvaluationConfig


class AudioContinuationEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.tokenizer = DACAudioTokenizer(
            num_quantizers=7,
            device=self.device,
        )

        token_rate = self.tokenizer.sampling_rate / self.tokenizer.frame_size
        self.token_rate = token_rate
        print(f"Token rate: {token_rate:.2f} tokens/sec")

        self.past_len = int(config.past_seconds * token_rate)
        self.future_len = int(config.predict_seconds * token_rate)
        print(f"Past length: {self.past_len} tokens ({config.past_seconds}s)")
        print(f"Future length: {self.future_len} tokens ({config.predict_seconds}s)")

        self.model = self._load_model()
        self.model.eval()

    def _load_model(self) -> AudioContinuationTransformer:
        config_dict = {
            "past_len": self.past_len,
            "future_len": self.future_len,
            "vocab_size": 1024,
            "n_codebooks": 7,
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 3,
            "d_ff": 256,
            "dropout": 0.1,
        }

        if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
            print(f"Loading checkpoint: {self.config.checkpoint_path}")
            checkpoint = torch.load(
                self.config.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            model = AudioContinuationTransformer(checkpoint.get("config", config_dict))
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("No checkpoint provided, using untrained model with random weights")
            model = AudioContinuationTransformer(config_dict)

        model.to(self.device)
        return model

    def _tokenize_audio(self, audio_path: str) -> torch.Tensor:
        print(f"Loading audio: {audio_path}")

        waveform, sr = self.tokenizer.load_audio_from_path(audio_path)
        waveform = waveform.to(self.device)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        tokens_list = []
        with torch.no_grad():
            for i in range(waveform.shape[0]):
                single_audio = waveform[i : i + 1]
                codes = self.tokenizer.encode_from_waveform(
                    single_audio, original_sampling_rate=sr
                )
                tokens_list.append(codes)

        tokens = torch.cat(tokens_list, dim=0)
        return tokens

    def _compute_metrics(
        self,
        pred_tokens: torch.Tensor,
        true_tokens: torch.Tensor,
    ) -> Dict[str, Any]:
        loss_fn = nn.CrossEntropyLoss(reduction="none")

        per_codebook_loss = []
        per_codebook_accuracy = []

        for cb_idx in range(self.model.n_codebooks):
            cb_logits = pred_tokens[:, :, cb_idx, :]
            cb_true = true_tokens[:, :, cb_idx]

            flat_logits = cb_logits.reshape(-1, self.model.vocab_size)
            flat_true = cb_true.reshape(-1)

            loss = loss_fn(flat_logits, flat_true)
            per_codebook_loss.append(loss.mean().item())

            preds = cb_logits.argmax(dim=-1)
            accuracy = (preds == cb_true).float().mean().item()
            per_codebook_accuracy.append(accuracy)

        total_loss = sum(
            loss * (0.7**i) for i, loss in enumerate(per_codebook_loss)
        ) / sum(0.7**i for i in range(self.model.n_codebooks))

        total_accuracy = sum(per_codebook_accuracy) / len(per_codebook_accuracy)

        return {
            "loss": total_loss,
            "accuracy": total_accuracy,
            "per_codebook_loss": per_codebook_loss,
            "per_codebook_accuracy": per_codebook_accuracy,
        }

    def _predict(
        self,
        prompt_tokens: torch.Tensor,
    ) -> torch.Tensor:
        prompt_tokens = prompt_tokens.to(self.device)

        generated = prompt_tokens.clone()
        max_new_tokens = self.future_len

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model.forward(generated)
                last_logits = logits[:, -1, :, :]

                next_tokens = []
                for cb_idx in range(self.model.n_codebooks):
                    cb_logits = last_logits[:, cb_idx, :].clone()

                    if self.config.repetition_penalty != 1.0:
                        past_tokens = generated[:, :, cb_idx]
                        for b in range(past_tokens.shape[0]):
                            unique_tokens = torch.unique(past_tokens[b])
                            cb_logits[b, unique_tokens] /= (
                                self.config.repetition_penalty
                            )

                    cb_logits = cb_logits / self.config.temperature

                    if self.config.top_k is not None and self.config.top_k > 0:
                        top_k_vals = torch.topk(
                            cb_logits, min(self.config.top_k, cb_logits.size(-1))
                        )
                        cb_logits[cb_logits < top_k_vals.values[:, -1:]] = float("-inf")

                    if self.config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            cb_logits, descending=True
                        )
                        probs_sorted = nn.functional.softmax(sorted_logits, dim=-1)
                        cumsum_probs = torch.cumsum(probs_sorted, dim=-1)

                        sorted_indices_to_remove = cumsum_probs > self.config.top_p
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
                generated = torch.cat([generated, next_frame], dim=1)

        return generated[:, prompt_tokens.shape[1] :, :]

    def evaluate(self, audio_file: Optional[str] = None) -> Dict[str, Any]:
        audio_file = audio_file or self.config.audio_file
        audio_path = os.path.join(self.config.audio_path, audio_file)

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {audio_file}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        tokens = self._tokenize_audio(audio_path)
        print(f"Total tokens: {tokens.shape}")

        n_codebooks, total_time = tokens.shape[1], tokens.shape[2]
        step_tokens = int(self.config.step_seconds * self.token_rate)

        num_windows = (total_time - self.past_len - self.future_len) // step_tokens
        if num_windows <= 0:
            raise ValueError(
                f"Audio too short: need at least {self.past_len + self.future_len} tokens, "
                f"got {total_time}"
            )

        if self.config.max_windows is not None:
            num_windows = min(num_windows, self.config.max_windows)

        print(f"Number of windows: {num_windows}")

        per_window_results = []
        all_losses = []
        all_accuracies = []
        all_inference_times = []
        all_per_codebook_loss = [[] for _ in range(self.model.n_codebooks)]
        all_per_codebook_accuracy = [[] for _ in range(self.model.n_codebooks)]

        tokens = tokens.transpose(1, 2).long()

        for window_idx in range(num_windows):
            start_idx = window_idx * step_tokens
            position_seconds = start_idx / self.token_rate

            past_tokens = tokens[:, start_idx : start_idx + self.past_len, :]

            true_future = tokens[
                :,
                start_idx + self.past_len : start_idx + self.past_len + self.future_len,
                :,
            ]

            inference_start = time.perf_counter()
            pred_future = self._predict(past_tokens)
            inference_end = time.perf_counter()
            inference_ms = (inference_end - inference_start) * 1000

            combined = torch.cat([past_tokens, pred_future], dim=1)
            with torch.no_grad():
                logits = self.model.forward(past_tokens, pred_future)

            metrics = self._compute_metrics(logits, true_future)

            all_losses.append(metrics["loss"])
            all_accuracies.append(metrics["accuracy"])
            all_inference_times.append(inference_ms)

            for cb_idx in range(self.model.n_codebooks):
                all_per_codebook_loss[cb_idx].append(
                    metrics["per_codebook_loss"][cb_idx]
                )
                all_per_codebook_accuracy[cb_idx].append(
                    metrics["per_codebook_accuracy"][cb_idx]
                )

            per_window_results.append(
                {
                    "window_idx": window_idx,
                    "position_sec": round(position_seconds, 2),
                    "loss": round(metrics["loss"], 4),
                    "accuracy": round(metrics["accuracy"], 4),
                    "per_codebook_loss": [
                        round(x, 4) for x in metrics["per_codebook_loss"]
                    ],
                    "per_codebook_accuracy": [
                        round(x, 4) for x in metrics["per_codebook_accuracy"]
                    ],
                    "inference_time_ms": round(inference_ms, 2),
                }
            )

            if (window_idx + 1) % 10 == 0:
                print(
                    f"Window {window_idx + 1}/{num_windows} | "
                    f"Pos: {position_seconds:.1f}s | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Acc: {metrics['accuracy']:.4f} | "
                    f"Time: {inference_ms:.0f}ms"
                )

        total_time_elapsed = time.time() - start_time

        avg_per_codebook_loss = [
            sum(vals) / len(vals) for vals in all_per_codebook_loss
        ]
        avg_per_codebook_accuracy = [
            sum(vals) / len(vals) for vals in all_per_codebook_accuracy
        ]

        results = {
            "config": {
                "checkpoint_path": self.config.checkpoint_path,
                "audio_file": audio_file,
                "step_seconds": self.config.step_seconds,
                "predict_seconds": self.config.predict_seconds,
                "past_seconds": self.config.past_seconds,
                "temperature": self.config.temperature,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
                "repetition_penalty": self.config.repetition_penalty,
                "device": self.config.device,
            },
            "summary": {
                "total_windows": num_windows,
                "audio_duration_sec": round(total_time / self.token_rate, 2),
                "total_evaluation_time_sec": round(total_time_elapsed, 2),
                "mean_loss": round(sum(all_losses) / len(all_losses), 4),
                "mean_accuracy": round(sum(all_accuracies) / len(all_accuracies), 4),
                "mean_inference_time_ms": round(
                    sum(all_inference_times) / len(all_inference_times), 2
                ),
                "min_inference_time_ms": round(min(all_inference_times), 2),
                "max_inference_time_ms": round(max(all_inference_times), 2),
                "per_codebook_loss": [round(x, 4) for x in avg_per_codebook_loss],
                "per_codebook_accuracy": [
                    round(x, 4) for x in avg_per_codebook_accuracy
                ],
            },
            "per_window": per_window_results,
        }

        return results

    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path
