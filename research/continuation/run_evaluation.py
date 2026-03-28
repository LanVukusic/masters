import sys
import os
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from research.continuation.config import EvaluationConfig
from research.continuation.audio_evaluator import AudioContinuationEvaluator


if __name__ == "__main__":
    config = EvaluationConfig(
        checkpoint_path=None,
        audio_path="dataset_gen/free_music/mp3s/",
        audio_file="100066 Lindstrom - Monsteer (Original Mix).mp3",
        step_seconds=0.5,
        predict_seconds=1.0,
        past_seconds=2.0,
        max_windows=20,
        output_dir="research/continuation/results",
    )

    evaluator = AudioContinuationEvaluator(config)

    results = evaluator.evaluate()

    evaluator.save_results(results, filename="evaluation_results.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total windows: {results['summary']['total_windows']}")
    print(f"Mean loss: {results['summary']['mean_loss']:.4f}")
    print(f"Mean accuracy: {results['summary']['mean_accuracy']:.4f}")
    print(f"Mean inference time: {results['summary']['mean_inference_time_ms']:.2f}ms")
    print("\nPer-codebook loss:")
    for i, loss in enumerate(results["summary"]["per_codebook_loss"]):
        print(f"  Codebook {i}: {loss:.4f}")
    print("\nPer-codebook accuracy:")
    for i, acc in enumerate(results["summary"]["per_codebook_accuracy"]):
        print(f"  Codebook {i}: {acc:.4f}")
