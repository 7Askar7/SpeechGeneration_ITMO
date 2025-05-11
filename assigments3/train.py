import os
import csv
from typing import List, Tuple, Dict

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import jiwer
from collections import defaultdict

# -------------------------
# Config & Vocabulary
# -------------------------
SR_TARGET = 16000
N_MELS = 80
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 30

# Mapping digits to Russian words
NUM2WORDS: Dict[str, str] = {
    '0': 'ноль',
    '1': 'одна',
    '2': 'две',
    '3': 'три',
    '4': 'четыре',
    '5': 'пять',
    '6': 'шесть',
    '7': 'семь',
    '8': 'восемь',
    '9': 'девять'
}

# -------------------------
# Normalization / Denormalization
# -------------------------

def normalize_label(digits: str) -> List[str]:
    """
    Преобразует строку цифр в список «словных» токенов,
    вставляет 'тысяча' и дополняет остаток до трёх цифр.
    """
    n = int(digits)
    thou = n // 1000
    rem = n % 1000
    tokens: List[str] = []
    if thou > 0:
        for d in str(thou):
            tokens.append(NUM2WORDS[d])
        tokens.append('тысяча')
        rem_str = str(rem).zfill(3)
        for d in rem_str:
            tokens.append(NUM2WORDS[d])
    else:
        for d in str(rem):
            tokens.append(NUM2WORDS[d])
    return tokens


def denormalize_pred(tokens: List[str]) -> str:
    """
    Обратное: ['одна','тысяча','пять','ноль','ноль'] -> '1005'
    """
    # Словарь обратного маппинга (кроме 'тысяча')
    WORD2NUM = {v: k for k, v in NUM2WORDS.items()}
    digits = ''
    for t in tokens:
        if t == 'тысяча':
            continue
        if t in WORD2NUM:
            digits += WORD2NUM[t]
    # Убираем ведущие нули при преобразовании
    return str(int(digits)) if digits else '0'

# Build vocabulary: blank + unique word tokens + 'тысяча'
WORD_TOKENS = sorted(set(NUM2WORDS.values()))
VOCAB = ['<blank>'] + WORD_TOKENS + ['тысяча']

token2idx = {tok: i for i, tok in enumerate(VOCAB)}
idx2token = {i: tok for tok, i in token2idx.items()}

# -------------------------
# Dataset & DataLoader
# -------------------------
class NumericSpeechDataset(Dataset):
    def __init__(self, csv_path: str, audio_dir: str):
        self.entries = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append(row)
        self.audio_dir = audio_dir
        # Fixed feature transforms
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR_TARGET,
            n_mels=N_MELS
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.entries[idx]
        filepath = os.path.join(self.audio_dir, row['filename'])
        waveform, sr = torchaudio.load(filepath)

        # Dynamic resampling
        if sr != SR_TARGET:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SR_TARGET)
            waveform = resampler(waveform)

        # Feature extraction
        mel = self.mel_spec(waveform)
        log_mel = self.db_transform(mel)
        features = log_mel.squeeze(0).transpose(0, 1)

        # Normalize transcription to word tokens
        raw = row['transcription']
        tokens = normalize_label(raw)
        token_ids = [token2idx[t] for t in tokens]
        target = torch.LongTensor(token_ids)

        return features, target, row['spk_id'], raw


def collate_fn(batch):
    feats, targets, spk_ids, raws = zip(*batch)
    feat_lengths = [f.size(0) for f in feats]
    tgt_lengths = [t.size(0) for t in targets]

    # Pad features
    max_feat = max(feat_lengths)
    feat_padded = torch.zeros(len(feats), max_feat, N_MELS)
    for i, f in enumerate(feats):
        feat_padded[i, :f.size(0), :] = f

    # Pad targets with blank token
    max_tgt = max(tgt_lengths)
    tgt_padded = torch.full((len(targets), max_tgt), fill_value=token2idx['<blank>'], dtype=torch.long)
    for i, t in enumerate(targets):
        tgt_padded[i, :t.size(0)] = t

    return feat_padded, torch.tensor(feat_lengths), tgt_padded, torch.tensor(tgt_lengths), list(spk_ids), list(raws)

# -------------------------
# Model Definition
# -------------------------
class CTCModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.rnn = nn.GRU(input_size=256, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.transpose(1, 2)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        # adjust lengths for conv downsampling (/4)
        down_len = ((lengths + 1) // 4).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, down_len, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(out)
        return logits.log_softmax(dim=-1)

# -------------------------
# Evaluation per speaker
# -------------------------
def evaluate_per_speaker(model: CTCModel, loader: DataLoader, device: torch.device):
    model.eval()
    cer_by_spk = defaultdict(list)
    with torch.no_grad():
        for feats, feat_lens, _, tgt_lens, spk_ids, raws in loader:
            feats = feats.to(device)
            log_probs = model(feats, feat_lens)  # [B, T, C]
            preds = log_probs.argmax(dim=-1).cpu().tolist()
            for pred_seq, raw in zip(preds, raws):
                # CTC greedy decode: remove repeats & blanks
                tokens_pred = []
                prev = None
                for idx in pred_seq:
                    if idx != token2idx['<blank>'] and idx != prev:
                        tokens_pred.append(idx2token[idx])
                    prev = idx
                pred_str = denormalize_pred(tokens_pred)
                cer = jiwer.cer(raw, pred_str)
                spk = spk_ids[preds.index(pred_seq)]
                cer_by_spk[spk].append(cer)
    # Print average CER per speaker
    print("Validation CER by speaker:")
    for spk, cers in cer_by_spk.items():
        avg = sum(cers) / len(cers)
        print(f"  {spk}: {avg:.3f}")

# -------------------------
# Training & Entry Point
# -------------------------
def train():
    train_csv, dev_csv = 'train/train.csv', 'dev/dev.csv'
    audio_dir = '.'

    train_ds = NumericSpeechDataset(train_csv, audio_dir)
    dev_ds = NumericSpeechDataset(dev_csv, audio_dir)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CTCModel(input_dim=N_MELS, hidden_dim=256, num_layers=3, num_classes=len(VOCAB)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ctc_loss = nn.CTCLoss(blank=token2idx['<blank>'], zero_infinity=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for feats, feat_lens, tgts, tgt_lens, _, _ in train_loader:
            feats, tgts = feats.to(device), tgts.to(device)
            optimizer.zero_grad()
            logits = model(feats, feat_lens)
            T = logits.size(1)
            loss = ctc_loss(
                logits.transpose(0, 1), tgts,
                input_lengths=torch.full((feats.size(0),), T, dtype=torch.long),
                target_lengths=tgt_lens
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}")

        # Validate per speaker
        evaluate_per_speaker(model, dev_loader, device)

    torch.save(model.state_dict(), 'ctc_model2.pth')
    print("Training complete. Model saved to ctc_model2.pth")

if __name__ == '__main__':
    train()
