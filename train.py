# we train a s2s model to predict the katakana phonemes from
# English phonemes
import argparse
import json
import random
from datetime import datetime
from functools import partial
from os import path
from random import randint

import torch
import torch.nn.functional as F
from g2p_en import G2p
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset, Sampler, random_split
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import BLEUScore

from hp import EOS_IDX, PAD_IDX, SOS_IDX, ascii_entries, en_phones, kanas


SEED = 3407
DIM = 256


class Model(nn.Module):
    def __init__(self, p2k: bool = False, dropout: float = 0.0):
        super().__init__()
        if p2k:
            self.e_emb = nn.Embedding(len(en_phones), DIM)
        else:
            self.e_emb = nn.Embedding(len(ascii_entries), DIM)
        self.k_emb = nn.Embedding(len(kanas), DIM)
        self.encoder = nn.GRU(DIM, DIM, batch_first=True, bidirectional=True)
        self.encoder_fc = nn.Sequential(
            nn.Linear(2 * DIM, DIM),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.pre_decoder = nn.GRU(DIM, DIM, batch_first=True)
        self.post_decoder = nn.GRU(2 * DIM, DIM, batch_first=True)
        self.attn = nn.MultiheadAttention(DIM, 4, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(DIM, len(kanas))
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # RNN 系は直交行列で初期化（学習安定化の定石）
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [B, Ts]
        tgt: [B, Tt]
        src_mask: [B, Ts]
        tgt_mask: [B, Tt]
        """
        e_emb = self.dropout(self.e_emb(src))
        k_emb = self.dropout(self.k_emb(tgt))
        k_emb = k_emb[:, :-1]
        enc_out, _ = self.encoder(e_emb)
        enc_out = self.encoder_fc(enc_out)
        dec_out, _ = self.pre_decoder(k_emb)
        attn_out, _ = self.attn.forward(dec_out, enc_out, enc_out, key_padding_mask=~src_mask)  # pyright: ignore[reportOptionalOperand]
        x = torch.cat([dec_out, attn_out], dim=-1)
        x, _ = self.post_decoder(x)
        x = self.fc(x)
        return x

    def inference(self, src):
        # Assume both src and tgt are unbatched
        sos_idx = SOS_IDX
        eos_idx = EOS_IDX
        src = src.unsqueeze(0)
        src_emb = self.e_emb(src)
        enc_out, _ = self.encoder(src_emb)
        enc_out = self.encoder_fc(enc_out)
        res = [sos_idx]
        h1 = None
        h2 = None
        count = 0
        while res[-1] != eos_idx and count < 16:
            dec = torch.tensor([res[-1]]).unsqueeze(0).to(src.device)
            dec_emb = self.k_emb(dec)
            dec_out, h1 = self.pre_decoder(dec_emb, h1)
            attn_out, _ = self.attn(dec_out, enc_out, enc_out)
            x = torch.cat([dec_out, attn_out], dim=-1)
            x, h2 = self.post_decoder(x, h2)
            x = self.fc(x)
            idx = torch.argmax(x, dim=-1)
            res.append(idx.cpu().item())  # pyright: ignore[reportArgumentType]
            count += 1
        return res


class MyDataset(Dataset):
    def __init__(self, path, device, p2k: bool = True):
        """
        reads a json line file
        """

        super().__init__()
        self.g2p = G2p()
        with open(path) as file:
            lines = file.readlines()
        self.data = [json.loads(line) for line in lines]
        self.device = device
        self.eng_dict = {c: i for i, c in enumerate(en_phones)}
        self.c_dict = {c: i for i, c in enumerate(ascii_entries)}
        self.kata_dict = {c: i for i, c in enumerate(kanas)}
        self.pad_idx = PAD_IDX
        self.sos_idx = SOS_IDX
        self.eos_idx = EOS_IDX
        self.cache_en = {}
        self.cache_kata = {}
        self.p2k_flag = p2k
        self.return_full = False

        # ローマ字/英語のインデックスを分離して保持
        # 2段階学習（Phased Training）で使用
        self.romaji_indices: list[int] = []
        self.english_indices: list[int] = []
        for idx, item in enumerate(self.data):
            if item.get("is_romaji", False):
                self.romaji_indices.append(idx)
            else:
                self.english_indices.append(idx)

        print(f"Dataset loaded: {len(self.english_indices)} English, {len(self.romaji_indices)} Romaji")

    def __len__(self):
        return len(self.data)

    def p2k(self, eng):
        phonemes = self.g2p(eng)
        # phonemes = [p[:-1] if p[-1] in "012" else p for p in phonemes]
        phonemes = list(filter(lambda x: x in self.eng_dict, phonemes))
        eng = [self.eng_dict[c] for c in phonemes]
        return eng

    def c2k(self, eng):
        eng = [self.c_dict[c] for c in eng]
        return eng

    def set_return_full(self, flag: bool):
        """
        Returns the full dataset, it's for bleu score calculation
        """
        self.return_full = flag

    def __getitem__(self, idx):
        if idx in self.cache_en:
            return self.cache_en[idx], self.cache_kata[idx]
        item = self.data[idx]
        eng = item["word"]
        katas = item["kata"]
        if self.p2k_flag:
            eng = self.p2k(eng)
        else:
            eng = self.c2k(eng)
        eng = [self.sos_idx, *eng, self.eos_idx]
        # katas is a list of katakana words
        # if not return_full, we randomly select one of them
        # else we return all of them
        if not self.return_full:
            kata = katas[randint(0, len(katas) - 1)]
            kata = [self.kata_dict[c] for c in kata]
            kata = [self.sos_idx, *kata, self.eos_idx]
            en = torch.tensor(eng).to(self.device)
            kana = torch.tensor(kata).to(self.device)
            self.cache_en[idx] = en
            self.cache_kata[idx] = kana
            return en, kana
        else:
            kata = []
            for k in katas:
                k = [self.kata_dict[c] for c in k]
                k = [self.sos_idx, *k, self.eos_idx]
                kata.append(torch.tensor(k).to(self.device))
            en = torch.tensor(eng).to(self.device)
            self.cache_en[idx] = en
            self.cache_kata[idx] = kata
            return en, kata


class PhasedSampler(Sampler):
    """
    2段階学習用サンプラー

    Phase 1: 英語データのみ（romaji_ratio = 0.0）
    Phase 2: 英語 + ローマ字（romaji_ratio を徐々に増加）

    ローマ字データはアップサンプリング（循環使用）され、英語データと混合される。
    これにより、モデルは英語の基盤を確立した後、ローマ字パターンを追加学習できる。

    Args:
        dataset: MyDataset インスタンス（english_indices, romaji_indices を持つ）
        romaji_ratio: 全体に対するローマ字サンプルの比率（0.0〜1.0）
    """

    def __init__(self, dataset, romaji_ratio: float = 0.0):
        self.english_indices = dataset.english_indices.copy()
        self.romaji_indices = dataset.romaji_indices.copy()
        self.romaji_ratio = romaji_ratio

    def set_romaji_ratio(self, ratio: float):
        """エポックごとにローマ字比率を更新"""

        self.romaji_ratio = min(1.0, max(0.0, ratio))

    def __iter__(self):
        # 英語データをシャッフル
        english_shuffled = self.english_indices.copy()
        random.shuffle(english_shuffled)

        # Phase 1: ローマ字比率が 0 または ローマ字データがない場合は英語のみ
        if self.romaji_ratio <= 0 or len(self.romaji_indices) == 0:
            return iter(english_shuffled)

        # Phase 2: 英語 + ローマ字の混合
        # ローマ字の必要数を計算（比率から逆算）
        # romaji_ratio = n_romaji / (n_english + n_romaji) を満たす n_romaji を求める
        n_english = len(english_shuffled)
        n_romaji_needed = int(n_english * self.romaji_ratio / (1 - self.romaji_ratio))

        # ローマ字をアップサンプリング（循環使用）
        romaji_shuffled = self.romaji_indices.copy()
        random.shuffle(romaji_shuffled)
        n_repeats = (n_romaji_needed // len(romaji_shuffled)) + 1
        romaji_upsampled = (romaji_shuffled * n_repeats)[:n_romaji_needed]
        random.shuffle(romaji_upsampled)

        # 混合してシャッフル
        combined = english_shuffled + romaji_upsampled
        random.shuffle(combined)

        return iter(combined)

    def __len__(self):
        if self.romaji_ratio <= 0:
            return len(self.english_indices)
        n_english = len(self.english_indices)
        n_romaji = int(n_english * self.romaji_ratio / (1 - self.romaji_ratio))
        return n_english + n_romaji


def get_romaji_ratio(epoch: int, phase1_epochs: int, final_ratio: float, rampup_epochs: int) -> float:
    """
    2段階学習におけるローマ字比率を計算する。

    Phase 1 (epoch <= phase1_epochs): 0.0（英語のみ）
    Phase 2 (epoch > phase1_epochs): 0.0 → final_ratio（徐々に増加）

    Args:
        epoch: 現在のエポック（1-indexed）
        phase1_epochs: Phase 1 のエポック数（英語のみ学習する期間）
        final_ratio: Phase 2 での最終的なローマ字比率
        rampup_epochs: ローマ字比率が 0 から final_ratio に達するまでのエポック数

    Returns:
        現在のエポックでのローマ字比率（0.0〜final_ratio）
    """

    # Phase 1: 英語のみ
    if epoch <= phase1_epochs:
        return 0.0

    # Phase 2: 徐々にローマ字を導入
    progress = min(1.0, (epoch - phase1_epochs) / rampup_epochs)
    return final_ratio * progress


def lens2mask(lens, max_len):
    mask = torch.zeros(len(lens), max_len).bool()
    for i, le in enumerate(lens):
        mask[i, :le] = True
    return mask


def collate_fn(batch, device):
    engs = [x[0] for x in batch]
    katas = [x[1] for x in batch]
    eng_lens = [len(x) for x in engs]
    kata_lens = [len(x) for x in katas]
    eng_mask = lens2mask(eng_lens, max(eng_lens))
    kata_mask = lens2mask(kata_lens, max(kata_lens))
    engs = pad_sequence(engs, batch_first=True, padding_value=0)
    katas = pad_sequence(katas, batch_first=True, padding_value=0)
    engs, katas, eng_mask, kata_mask = [x.to(device) for x in [engs, katas, eng_mask, kata_mask]]
    return engs, katas, eng_mask, kata_mask


def infer(src, model, p2k):
    model = model.eval()
    res = model.inference(src)
    # return to words
    res = [kanas[i] for i in res]
    # also for english phonemes
    if p2k:
        src = [en_phones[i] for i in src]
    else:
        src = [ascii_entries[i] for i in src]
    return src, res


def tensor2str(t):
    return " ".join([str(int(x)) for x in t])


def evaluate_by_task(
    model: Model,
    dataset: MyDataset,
    val_indices,
    device: torch.device,
    sample_size: int = 1000,
) -> dict:
    """
    英語タスクとローマ字タスクを分離して評価する。

    各タスクの BLEU スコアを個別に計算し、モデルのタスク別性能を監視する。
    これにより、Phase 2 でローマ字を導入した際の英語性能への影響を検知できる。

    Args:
        model: 評価対象のモデル
        dataset: 評価用データセット（is_romaji 情報を含む）
        val_indices: バリデーション用インデックスのリスト
        device: 使用するデバイス
        sample_size: 評価に使用するサンプル数

    Returns:
        dict: {
            'english_bleu': float,  # 英語タスクの BLEU スコア
            'romaji_bleu': float,   # ローマ字タスクの BLEU スコア
            'overall_bleu': float,  # 全体の BLEU スコア
            'english_count': int,   # 評価した英語サンプル数
            'romaji_count': int,    # 評価したローマ字サンプル数
        }
    """

    bleu_english = BLEUScore(n_gram=3)
    bleu_romaji = BLEUScore(n_gram=3)
    bleu_overall = BLEUScore(n_gram=3)

    english_count = 0
    romaji_count = 0

    # シード固定でサンプリング（再現性のため）
    torch.manual_seed(SEED)
    actual_sample_size = min(sample_size, len(val_indices))
    sample_indices = torch.randperm(len(val_indices))[:actual_sample_size].tolist()

    dataset.set_return_full(True)
    model.eval()

    with torch.no_grad():
        for idx in sample_indices:
            original_idx = val_indices[idx]
            eng, kata_list = dataset[original_idx]
            item = dataset.data[original_idx]
            is_romaji = item.get("is_romaji", False)

            res = model.inference(eng)
            pred_kana = tensor2str(res)
            kana = [[tensor2str(k) for k in kata_list]]

            bleu_overall.update(pred_kana, kana)

            if is_romaji:
                bleu_romaji.update(pred_kana, kana)
                romaji_count += 1
            else:
                bleu_english.update(pred_kana, kana)
                english_count += 1

    dataset.set_return_full(False)

    return {
        "english_bleu": bleu_english.compute().item() if english_count > 0 else 0.0,
        "romaji_bleu": bleu_romaji.compute().item() if romaji_count > 0 else 0.0,
        "overall_bleu": bleu_overall.compute().item(),
        "english_count": english_count,
        "romaji_count": romaji_count,
    }


def get_timestamp_suffix() -> str:
    """
    現在の日付時刻を MMDDHHMM 形式で取得する
    例: 11月24日13時30分 -> _11241330
    """
    now = datetime.now()
    return f"_{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}"


def get_unique_filepath(base_dir: str, base_name: str, extension: str = "pth") -> str:
    """
    既存ファイルと衝突しないユニークなファイルパスを生成する
    日付時刻の suffix を追加し、それでも衝突する場合は追加の suffix を付与する

    Args:
        base_dir: 保存先ディレクトリ
        base_name: ベースファイル名（拡張子なし）
        extension: ファイル拡張子（デフォルト: pth）

    Returns:
        ユニークなファイルパス
    """
    timestamp = get_timestamp_suffix()
    base_path = path.join(base_dir, f"{base_name}{timestamp}.{extension}")

    # ファイルが存在しない場合はそのまま返す
    if not path.exists(base_path):
        return base_path

    # 衝突する場合は suffix を追加
    suffix_num = 1
    while True:
        candidate_path = path.join(base_dir, f"{base_name}{timestamp}_{suffix_num}.{extension}")
        if not path.exists(candidate_path):
            return candidate_path
        suffix_num += 1


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    知識蒸留用の損失関数
    T: Temperature (温度)
    alpha: 蒸留損失の重み (0.0 - 1.0)
    """
    # KL Divergence Loss (先生の分布を真似る)
    # F.log_softmax(student) と F.softmax(teacher) を比較
    dist_loss = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / T, dim=-1), F.softmax(teacher_logits / T, dim=-1)
    ) * (T * T)

    # 通常の CrossEntropy Loss (正解ラベルを学習)
    ce_loss = F.cross_entropy(student_logits.transpose(1, 2), labels, ignore_index=0)

    return alpha * dist_loss + (1 - alpha) * ce_loss


def train():
    torch.manual_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="vendor/katakana_dict.jsonl")
    parser.add_argument("--p2k", action="store_true")
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
        help="Model dimension (default: 256, optimized for speed and accuracy balance)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs (default: 50, sufficient for ExponentialLR)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64, small batch helps generalization for small models)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience (default: 10, increased for longer training)"
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping threshold (default: 1.0)")
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate (default: 0.0, maximize capacity for dim=256)"
    )

    # 知識蒸留用の引数
    parser.add_argument("--teacher", type=str, default=None, help="Path to teacher model for knowledge distillation")
    parser.add_argument("--distill-temp", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="Weight for distillation loss (0.0-1.0)")

    # 2段階学習（Phased Training）用の引数
    # Phase 1: 英語のみで学習 → Phase 2: ローマ字を徐々に混合
    # デフォルト値は最適化済み
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=10,
        help="Number of epochs for Phase 1 (English only). Set to 0 to disable phased training (default: 10)",
    )
    parser.add_argument(
        "--romaji-ratio",
        type=float,
        default=0.2,
        help="Final romaji ratio in Phase 2 (default: 0.2 = 20%% of batch)",
    )
    parser.add_argument(
        "--rampup-epochs",
        type=int,
        default=10,
        help="Number of epochs to ramp up romaji ratio from 0 to final (default: 10)",
    )

    parser.add_argument(
        "--inference-dim",
        type=int,
        default=None,
        help="Separate model dimension for inference export (default: None, uses --dim). Set to 256 for faster inference.",
    )
    args = parser.parse_args()

    global DIM
    DIM = args.dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(p2k=args.p2k, dropout=args.dropout).to(device)

    # Teacher モデルの準備
    teacher_model = None
    if args.teacher:
        print(f"Loading teacher model from {args.teacher}...")
        # Teacher モデルの次元数を自動検出
        state_dict = torch.load(args.teacher, map_location="cpu")
        if "e_emb.weight" in state_dict:
            teacher_dim = state_dict["e_emb.weight"].shape[1]
        else:
            teacher_dim = 512  # fallback

        # DIM を一時的に変更してTeacherモデルを作成
        current_dim = DIM
        DIM = teacher_dim
        teacher_model = Model(p2k=args.p2k).to(device)
        teacher_model.load_state_dict(state_dict)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        DIM = current_dim
        print(f"Teacher model loaded (dim={teacher_dim}). Distilling to student (dim={DIM}).")

    dataset = MyDataset(args.data, device, p2k=args.p2k)
    train_ds, val_ds = random_split(dataset, [0.95, 0.05])

    # バリデーション用に return_full を有効化したデータセットを作成（別インスタンス）
    # val_ds は Subset なので、元のデータセットとインデックスを取得
    val_dataset_full = MyDataset(args.data, device, p2k=args.p2k)
    val_indices = val_ds.indices

    # 2段階学習（Phased Training）の設定
    # phase1_epochs > 0 の場合は PhasedSampler を使用
    is_phased_training = args.phase1_epochs > 0
    phased_sampler = None

    if is_phased_training:
        # PhasedSampler を使用（Phase 1 は英語のみ、Phase 2 はローマ字を徐々に混合）
        # train_ds は Subset なので、元のデータセットからサンプラーを作成
        # Subset のインデックスを使ってサンプラーのインデックスをマッピングする必要がある
        train_indices = train_ds.indices

        # 元データセットのインデックスから train_ds 用のインデックスに変換
        train_english_indices = [i for i, idx in enumerate(train_indices) if idx in set(dataset.english_indices)]
        train_romaji_indices = [i for i, idx in enumerate(train_indices) if idx in set(dataset.romaji_indices)]

        # PhasedSampler 用のダミーデータセットオブジェクト
        class _SamplerDataset:
            def __init__(self, english_indices, romaji_indices):
                self.english_indices = english_indices
                self.romaji_indices = romaji_indices

        sampler_dataset = _SamplerDataset(train_english_indices, train_romaji_indices)
        phased_sampler = PhasedSampler(sampler_dataset, romaji_ratio=0.0)

        print(f"Phased training enabled: Phase 1 = {args.phase1_epochs} epochs (English only)")
        print(f"  Phase 2: romaji ratio 0 -> {args.romaji_ratio:.0%} over {args.rampup_epochs} epochs")
        print(f"  Train set: {len(train_english_indices)} English, {len(train_romaji_indices)} Romaji")

        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=phased_sampler,
            collate_fn=partial(collate_fn, device=device),
        )
    else:
        # 従来の動作（shuffle=True）
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, device=device),
        )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_fn, device=device),
    )

    # Label Smoothing: Removed to allow model to fully trust the high-quality dataset
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # Optimizer: AdamW with 0 weight decay to allow full memorization for small model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Scheduler: ExponentialLR enforces convergence (Simulating the original aggressive strategy)
    # gamma=0.9 means LR decays to ~35% after 10 epochs, ~12% after 20 epochs.
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    writer = SummaryWriter()

    best_val_loss = float("inf")
    best_bleu_score = 0.0
    patience_counter = 0
    best_model_state = None

    steps = 0
    phase2_started = False
    for epoch in range(1, args.epochs + 1):
        # 2段階学習: エポックごとにローマ字比率を更新
        if is_phased_training and phased_sampler is not None:
            romaji_ratio = get_romaji_ratio(
                epoch,
                args.phase1_epochs,
                args.romaji_ratio,
                args.rampup_epochs,
            )
            phased_sampler.set_romaji_ratio(romaji_ratio)

            # Phase 2 開始時にスケジューラーを調整（より緩やかな減衰に切り替え）
            if epoch == args.phase1_epochs + 1 and not phase2_started:
                phase2_started = True
                print("=== Phase 2 started: mixing romaji data (gamma=0.95) ===")
                scheduler = ExponentialLR(optimizer, gamma=0.95)

        model.train()
        epoch_train_loss = 0.0
        train_batches = 0
        for eng, kata, e_mask, k_mask in train_dl:
            optimizer.zero_grad()
            out = model(eng, kata, e_mask, k_mask)

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_out = teacher_model(eng, kata, e_mask, k_mask)
                # 知識蒸留 Loss
                loss = distillation_loss(out, teacher_out, kata[:, 1:], args.distill_temp, args.distill_alpha)
            else:
                # 通常の Loss
                loss = criterion(out.transpose(1, 2), kata[:, 1:])

            writer.add_scalar("Loss/train", loss.item(), steps)
            epoch_train_loss += loss.item()
            train_batches += 1
            loss.backward()
            # 勾配クリッピングを追加
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            steps += 1

        model.eval()
        total_loss = 0
        count = 0
        with torch.no_grad():
            for eng, kata, e_mask, k_mask in val_dl:
                out = model(eng, kata, e_mask, k_mask)
                loss = criterion(out.transpose(1, 2), kata[:, 1:])
                total_loss += loss.item()
                count += 1

        val_loss = total_loss / count
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/train_epoch", epoch_train_loss / train_batches, epoch)

        # BLEU スコアを計算（サンプリングして計算）
        # 2段階学習が有効な場合はタスク別評価を実施
        if is_phased_training:
            # タスク別評価（英語/ローマ字を分離して評価）
            metrics = evaluate_by_task(model, val_dataset_full, val_indices, device)
            bleu_score = metrics["overall_bleu"]
            english_bleu = metrics["english_bleu"]
            romaji_bleu = metrics["romaji_bleu"]

            writer.add_scalar("BLEU/val", bleu_score, epoch)
            writer.add_scalar("BLEU/english", english_bleu, epoch)
            writer.add_scalar("BLEU/romaji", romaji_bleu, epoch)
            writer.add_scalar("Curriculum/romaji_ratio", romaji_ratio, epoch)
        else:
            # 従来の評価（全体 BLEU のみ）
            # 公平な比較のため、シードを固定してサンプリング
            bleu = BLEUScore(n_gram=3)
            sample_size = min(1000, len(val_ds))
            # シードを固定して同じサンプルを毎回評価
            torch.manual_seed(SEED)
            sample_indices = torch.randperm(len(val_ds))[:sample_size].tolist()
            # return_full を一時的に有効化
            val_dataset_full.set_return_full(True)
            with torch.no_grad():
                for idx in sample_indices:
                    original_idx = val_indices[idx]
                    eng, kata_list = val_dataset_full[original_idx]
                    res = model.inference(eng)
                    pred_kana = tensor2str(res)
                    kana = [[tensor2str(k) for k in kata_list]]
                    bleu.update(pred_kana, kana)
            val_dataset_full.set_return_full(False)
            bleu_score = bleu.compute().item()
            writer.add_scalar("BLEU/val", bleu_score, epoch)
            english_bleu = None
            romaji_bleu = None

        # 学習率スケジューラーを更新
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LearningRate", current_lr, epoch)

        # サンプルを表示
        sample = val_ds[randint(0, len(val_ds) - 1)]
        src, _ = sample  # pyright: ignore[reportGeneralTypeIssues]
        src, pred = infer(src, model, args.p2k)
        print(f"Epoch {epoch} Sample: {src} -> {pred}")

        # 2段階学習の場合は詳細な情報を表示
        if is_phased_training:
            phase_str = (
                "Phase 1 (English only)" if epoch <= args.phase1_epochs else f"Phase 2 (Romaji: {romaji_ratio:.1%})"
            )
            print(
                f"Epoch {epoch} [{phase_str}] Train Loss: {epoch_train_loss / train_batches:.4f}, Val Loss: {val_loss:.4f}"
            )
            print(
                f"  Overall BLEU: {bleu_score:.4f}, English BLEU: {english_bleu:.4f}, Romaji BLEU: {romaji_bleu:.4f}, LR: {current_lr:.2e}"
            )

            # 英語 BLEU が大幅に低下したら警告
            if epoch > args.phase1_epochs and english_bleu is not None and english_bleu < 0.92:
                print("  Warning: English BLEU dropped below 0.92!")
        else:
            print(
                f"Epoch {epoch} Train Loss: {epoch_train_loss / train_batches:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu_score:.4f}, LR: {current_lr:.2e}"
            )

        # 学習率が変更された場合は表示
        if old_lr != current_lr:
            print(f"  -> Learning rate reduced: {old_lr:.2e} -> {current_lr:.2e}")

        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_bleu_score = bleu_score
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  -> New best model! Val Loss: {val_loss:.4f}, BLEU: {bleu_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  -> Early stopping triggered after {args.patience} epochs without improvement")
                break

        name = "p2k" if args.p2k else "c2k"

    # 最良モデルを保存
    if best_model_state is not None:
        best_model_path = get_unique_filepath("vendor", f"model-{name}-best")
        torch.save(
            best_model_state,
            best_model_path,
        )
        print(f"Best model saved to {best_model_path}: Val Loss: {best_val_loss:.4f}, BLEU: {best_bleu_score:.4f}")

    # 最後のエポックのモデルも保存
    final_model_path = get_unique_filepath("vendor", f"model-{name}-e{epoch}")
    torch.save(
        model.state_dict(),
        final_model_path,
    )
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    train()
