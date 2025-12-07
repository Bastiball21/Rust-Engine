import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct
import os

# --- ARCHITECTURE CONFIG ---
# Must match nnue.rs constants
LAYER1_SIZE = 256
INPUT_SIZE = 41024 # HalfKP
HIDDEN_SIZE = 32
# Quantization params
QA = 255
QB = 64

# --- HELPER: HalfKP Indexing ---
def make_halfkp_index(perspective, king_sq, piece, sq):
    orient_sq = sq if perspective == 0 else sq ^ 56
    orient_king = king_sq if perspective == 0 else king_sq ^ 56
    piece_color = 0 if piece < 6 else 1
    piece_type = piece % 6
    if piece_type == 5: return None
    kp_idx = piece_type if piece_color == perspective else piece_type + 5
    return orient_king * 641 + kp_idx * 64 + orient_sq

# --- DATASET ---
class ChessDataset(Dataset):
    def __init__(self, filename):
        self.samples = []
        print(f"Loading data from {filename}...")
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 3: continue
                fen = parts[0].strip()
                score = float(parts[1].strip())
                result = float(parts[2].strip())
                self.samples.append((fen, score, result))
        print(f"Loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, score, result = self.samples[idx]
        return self.fen_to_features(fen), torch.tensor([result], dtype=torch.float32)

    def fen_to_features(self, fen):
        parts = fen.split()
        board_str = parts[0]
        turn = 0 if parts[1] == 'w' else 1
        pieces = []
        king_sq = [0, 0]
        rank = 7
        file = 0
        for char in board_str:
            if char == '/':
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)
            else:
                sq = rank * 8 + file
                p_map = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,
                         'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
                p_idx = p_map[char]
                pieces.append((p_idx, sq))
                if p_idx == 5: king_sq[0] = sq
                if p_idx == 11: king_sq[1] = sq
                file += 1

        stm_perspective = turn
        stm_king = king_sq[stm_perspective]
        nstm_perspective = 1 - turn
        nstm_king = king_sq[nstm_perspective]

        indices_stm = []
        indices_nstm = []
        for p_idx, sq in pieces:
            idx = make_halfkp_index(stm_perspective, stm_king, p_idx, sq)
            if idx is not None: indices_stm.append(idx)
            idx = make_halfkp_index(nstm_perspective, nstm_king, p_idx, sq)
            if idx is not None: indices_nstm.append(idx)

        return torch.tensor(indices_stm, dtype=torch.long), torch.tensor(indices_nstm, dtype=torch.long)

def collate_fn(batch):
    targets = torch.stack([item[1] for item in batch])
    MAX_INDICES = 32
    stm_batch = torch.zeros(len(batch), MAX_INDICES, dtype=torch.long)
    nstm_batch = torch.zeros(len(batch), MAX_INDICES, dtype=torch.long)
    for i, ((stm, nstm), _) in enumerate(batch):
        len_s = min(len(stm), MAX_INDICES)
        stm_batch[i, :len_s] = stm[:len_s]
        len_n = min(len(nstm), MAX_INDICES)
        nstm_batch[i, :len_n] = nstm[:len_n]
    return stm_batch, nstm_batch, targets

class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.feature_weights = nn.EmbeddingBag(INPUT_SIZE, LAYER1_SIZE, mode='sum')
        self.feature_bias = nn.Parameter(torch.zeros(LAYER1_SIZE))
        nn.init.normal_(self.feature_weights.weight, std=0.01)
        self.l1 = nn.Linear(2 * LAYER1_SIZE, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, stm_indices, nstm_indices):
        stm_acc = self.feature_weights(stm_indices) + self.feature_bias
        nstm_acc = self.feature_weights(nstm_indices) + self.feature_bias
        stm_acc = torch.clamp(stm_acc, 0.0, 1.0)
        nstm_acc = torch.clamp(nstm_acc, 0.0, 1.0)
        x = torch.cat([stm_acc, nstm_acc], dim=1)
        x = self.l1(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.l2(x)
        x = torch.clamp(x, 0.0, 1.0)
        x = self.output(x)
        return torch.sigmoid(x)

# --- EXPORTER ---
def export_net(model, filename="nn-aether.nnue"):
    print(f"Exporting to {filename}...")

    # Weights are floats. We scale them to integers.
    # Feature Transformer
    ft_w = (model.feature_weights.weight.data * QA).round().to(torch.int16).cpu().numpy().flatten()
    ft_b = (model.feature_bias.data * QA).round().to(torch.int16).cpu().numpy().flatten()

    # Layer 1
    # Weight scale: QB
    # Bias scale: QA * QB (Accumulator * Weight = QA * QB)
    l1_w = (model.l1.weight.data * QB).round().clamp(-127, 127).to(torch.int8).cpu().numpy().flatten()
    l1_b = (model.l1.bias.data * (QA * QB)).round().to(torch.int32).cpu().numpy().flatten()

    # Layer 2
    # Input scale: QA (result of L1 >> 6)
    # Weight scale: QB
    # Bias scale: QA * QB
    l2_w = (model.l2.weight.data * QB).round().clamp(-127, 127).to(torch.int8).cpu().numpy().flatten()
    l2_b = (model.l2.bias.data * (QA * QB)).round().to(torch.int32).cpu().numpy().flatten()

    # Output Layer
    # Input scale: QA (result of L2 >> 6)
    # Weight scale: QB
    # Bias scale: QA * QB
    out_w = (model.output.weight.data * QB).round().clamp(-127, 127).to(torch.int8).cpu().numpy().flatten()
    out_b = (model.output.bias.data * (QA * QB)).round().to(torch.int32).cpu().numpy().flatten()

    with open(filename, "wb") as f:
        f.write(struct.pack("<I", 0x7AF32F2D))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

        f.write(ft_b.tobytes())
        f.write(ft_w.tobytes())
        f.write(struct.pack("<I", 0)) # HashNetwork
        f.write(l1_b.tobytes())
        f.write(l1_w.tobytes())
        f.write(l2_b.tobytes())
        f.write(l2_w.tobytes())
        f.write(out_b.tobytes())
        f.write(out_w.tobytes())

    print("Done.")

def train():
    if not os.path.exists("aether_data.txt"):
        print("Data file 'aether_data.txt' not found. Generating random net only.")
        model = NNUE()
        export_net(model)
        return

    dataset = ChessDataset("aether_data.txt")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = NNUE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Starting Training...")
    for epoch in range(1):
        total_loss = 0
        for stm, nstm, target in dataloader:
            optimizer.zero_grad()
            output = model(stm, nstm)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")

    export_net(model)

if __name__ == "__main__":
    train()
