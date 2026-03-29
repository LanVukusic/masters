import sys
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from research.continuation.config import EvaluationConfig
from research.continuation.audio_evaluator import AudioContinuationEvaluator


if __name__ == "__main__":
    songs = [
        "190288 Central Avenue - For Only You feat. Ov (Hard Mix Vocal Latin Mix).mp3",
        "1362355 Broady Champs - Hard!!!!!!!!! (Original Mix).mp3",
    ]

    for song in songs:
        print(f"\n{'#' * 60}")
        print(f"# Evaluating: {song}")
        print(f"{'#' * 60}\n")

        config = EvaluationConfig(
            checkpoint_path="checkpoints/audio_AUTOREG_29-0025_mid.pt",
            audio_path="dataset_gen/free_music/mp3s/",
            audio_file=song,
            step_seconds=0.5,
            predict_seconds=1.0,
            past_seconds=2.0,
            max_windows=20,
            output_dir="research/continuation/results",
        )

        evaluator = AudioContinuationEvaluator(config)
        results = evaluator.evaluate()

        safe_name = song.replace(" ", "_").replace("/", "_")[:50]
        evaluator.save_results(results, filename=f"eval_{safe_name}.json")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total windows: {results['summary']['total_windows']}")
        print(f"Mean loss: {results['summary']['mean_loss']:.4f}")
        print(f"Mean accuracy: {results['summary']['mean_accuracy']:.4f}")
        print(
            f"Mean inference time: {results['summary']['mean_inference_time_ms']:.2f}ms"
        )
        print("\nPer-codebook loss:")
        for i, loss in enumerate(results["summary"]["per_codebook_loss"]):
            print(f"  Codebook {i}: {loss:.4f}")
        print("\nPer-codebook accuracy:")
        for i, acc in enumerate(results["summary"]["per_codebook_accuracy"]):
            print(f"  Codebook {i}: {acc:.4f}")
