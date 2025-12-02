# Description: Evaluate the model on the full dataset.
# and calculate the accuracy.
import argparse

import torch
from torcheval.metrics import BLEUScore
from tqdm.auto import tqdm

import train
from train import Model, MyDataset, random_split


parser = argparse.ArgumentParser()

parser.add_argument("--data", type=str, default="vendor/katakana_dict.jsonl")
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--p2k", action="store_true")
parser.add_argument("--dim", type=int, default=None, help="Model dimension (auto-detected if not specified)")
parser.add_argument("--by-task", action="store_true", help="Evaluate English and Romaji separately")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの次元数を自動検出（重みから推定）
if args.dim is None:
    state_dict = torch.load(args.model, map_location="cpu")
    # embedding層の重みから次元数を推定
    if "e_emb.weight" in state_dict:
        dim = state_dict["e_emb.weight"].shape[1]
    elif "k_emb.weight" in state_dict:
        dim = state_dict["k_emb.weight"].shape[1]
    else:
        # デフォルト値（元のモデルは256）
        dim = 256
        print(f"Warning: Could not detect model dimension, using default: {dim}")
    print(f"Auto-detected model dimension: {dim}")
else:
    dim = args.dim

# グローバル変数を設定してからモデルを作成
train.DIM = dim
model = Model(p2k=args.p2k).to(device)

model.load_state_dict(torch.load(args.model, map_location=device))

model.eval()

torch.manual_seed(3407)

dataset = MyDataset(args.data, device, p2k=args.p2k)
test_ds, _ = random_split(dataset, [0.1, 0.9])
dataset.set_return_full(True)  # bleu score test


def tensor2str(t):
    return " ".join([str(int(x)) for x in t])


# タスク別評価が有効な場合は、英語とローマ字を分離して評価
if args.by_task:
    bleu_overall = BLEUScore(n_gram=3)
    bleu_english = BLEUScore(n_gram=3)
    bleu_romaji = BLEUScore(n_gram=3)

    english_count = 0
    romaji_count = 0

    for i in tqdm(range(len(test_ds))):
        original_idx = test_ds.indices[i]
        eng, kata = test_ds[i]  # pyright: ignore[reportGeneralTypeIssues]
        item = dataset.data[original_idx]
        is_romaji = item.get("is_romaji", False)

        res = model.inference(eng)
        pred_kana = tensor2str(res)
        kana = [[tensor2str(k) for k in kata]]

        bleu_overall.update(pred_kana, kana)

        if is_romaji:
            bleu_romaji.update(pred_kana, kana)
            romaji_count += 1
        else:
            bleu_english.update(pred_kana, kana)
            english_count += 1

    overall_score = bleu_overall.compute().item()
    english_score = bleu_english.compute().item() if english_count > 0 else 0.0
    romaji_score = bleu_romaji.compute().item() if romaji_count > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"Overall BLEU: {overall_score:.4f}")
    print(f"English BLEU: {english_score:.4f} ({english_count} samples)")
    print(f"Romaji BLEU:  {romaji_score:.4f} ({romaji_count} samples)")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Test samples: {len(test_ds)}")
    print(f"{'=' * 60}")
else:
    # 従来の評価（全体 BLEU のみ）
    bleu = BLEUScore(n_gram=3)

    for i in tqdm(range(len(test_ds))):
        eng, kata = test_ds[i]  # pyright: ignore[reportGeneralTypeIssues]
        res = model.inference(eng)
        pred_kana = tensor2str(res)
        kana = [[tensor2str(k) for k in kata]]
        bleu.update(pred_kana, kana)

    bleu_score = bleu.compute().item()
    print(f"\n{'=' * 60}")
    print(f"BLEU score: {bleu_score:.4f}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Test samples: {len(test_ds)}")
    print(f"{'=' * 60}")
