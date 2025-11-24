"""
推論速度のベンチマークスクリプト
モデルサイズが256→512に増えた場合の推論速度への影響を測定する
"""

import argparse
import time

from tqdm import tqdm

from src.e2k.inference import C2K, P2K


def benchmark_model(model, test_words: list[str], num_iterations: int = 1000):
    """
    モデルの推論速度をベンチマークする

    Args:
        model: C2K または P2K モデル
        test_words: テスト用の単語リスト
        num_iterations: 繰り返し回数

    Returns:
        (平均時間(秒), 1秒あたりの推論回数)
    """
    # ウォームアップ
    for word in test_words[:10]:
        _ = model(word)

    # ベンチマーク実行
    start_time = time.perf_counter()
    for _ in tqdm(range(num_iterations), desc="Benchmarking"):
        for word in test_words:
            _ = model(word)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_inferences = num_iterations * len(test_words)
    avg_time_per_inference = total_time / total_inferences
    inferences_per_second = total_inferences / total_time

    return avg_time_per_inference, inferences_per_second


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument("--model-type", choices=["c2k", "p2k"], default="c2k", help="Model type to benchmark")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .npz model file (optional)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--words", type=str, nargs="+", default=None, help="Custom test words")
    args = parser.parse_args()

    # テスト用の単語リスト（カタカナ変換の典型的な長さ）
    if args.words:
        test_words = args.words
    else:
        test_words = [
            "computer",
            "internet",
            "smartphone",
            "algorithm",
            "database",
            "software",
            "hardware",
            "network",
            "security",
            "application",
            "programming",
            "development",
            "framework",
            "library",
            "function",
        ]

    print(f"Benchmarking {args.model_type.upper()} model...")
    if args.model_path:
        print(f"Using custom model: {args.model_path}")

    print(f"Test words: {len(test_words)} words")
    print(f"Iterations: {args.iterations}")
    print(f"Total inferences: {args.iterations * len(test_words)}")
    print("-" * 60)

    # モデルをロード
    if args.model_path:
        # 指定されたパスからモデルをロード
        if args.model_type == "c2k":
            model = C2K(model_path=args.model_path)
        else:
            model = P2K(model_path=args.model_path)
    elif args.model_type == "c2k":
        model = C2K()
    else:
        model = P2K()

    # モデルサイズを推定（重みの形状から）
    dim = 0
    try:
        # encoderの重みから次元数を推定
        if hasattr(model, "s2s"):
            encoder_weight = model.s2s.encoder.cell.ih.weight
            if encoder_weight is not None:
                dim = encoder_weight.shape[-1]
                print(f"Estimated model dimension: {dim}")
    except AttributeError:
        print("Could not estimate model dimension (encoder weights not accessible)")

    # ベンチマーク実行
    avg_time, ips = benchmark_model(model, test_words, args.iterations)

    print("-" * 60)
    print("Results:")
    print(f"  Average time per inference: {avg_time * 1000:.3f} ms")
    print(f"  Inferences per second: {ips:.1f} inf/s")
    print(f"  Total time: {args.iterations * len(test_words) * avg_time:.2f} s")
    if dim == 512:
        print("\nNote: This is a 512-dim model.")
    elif dim == 256:
        print("\nNote: This is a 256-dim model.")


if __name__ == "__main__":
    main()
