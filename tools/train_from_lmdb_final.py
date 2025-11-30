import os
import sys
import json
import argparse
import math
from typing import List, Tuple

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.jit import ScriptModule


# ---------------------------
# 경로 설정
# ---------------------------

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))

DEFAULT_LMDB_ROOT = os.path.join(PROJECT_ROOT, "data", "lmdb", "standard_2025_01")
DEFAULT_SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "runs", "chessnet")


# ---------------------------
# LMDB Dataset
# ---------------------------

class LMDBShardInfo:
    def __init__(self, shard_index: int, shard_path: str, num_samples: int, offset: int):
        self.shard_index = shard_index
        self.shard_path = shard_path
        self.num_samples = num_samples
        self.offset = offset  # global index 시작점
        self.env = None       # lazy open


class LMDBDataset(Dataset):
    """
    parse_pgn.py가 만든 shard_* 구조를 그대로 읽는 Dataset.
    각 샘플:
      - tensor: uint8 bytes, shape (18,8,8)
      - label: {policy:int, value:float}
    """

    def __init__(self, lmdb_root: str):
        super().__init__()
        self.lmdb_root = lmdb_root
        self.shards: List[LMDBShardInfo] = []
        self.total_samples = 0

        self._scan_shards()

    def _scan_shards(self):
        if not os.path.isdir(self.lmdb_root):
            raise FileNotFoundError(f"lmdb_root not found: {self.lmdb_root}")

        shard_names = sorted(
            d for d in os.listdir(self.lmdb_root)
            if d.startswith("shard_") and os.path.isdir(os.path.join(self.lmdb_root, d))
        )

        offset = 0
        for name in shard_names:
            shard_dir = os.path.join(self.lmdb_root, name)
            meta_path = os.path.join(shard_dir, "meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            num_samples = int(meta.get("num_samples", 0))
            if num_samples <= 0:
                continue
            idx = int(name.split("_")[1])
            info = LMDBShardInfo(idx, shard_dir, num_samples, offset)
            self.shards.append(info)
            offset += num_samples

        self.total_samples = offset
        if self.total_samples == 0:
            raise RuntimeError(f"No samples found under {self.lmdb_root}")

        print(f"[LMDBDataset] {len(self.shards)} shards, total_samples={self.total_samples}")

    def __len__(self):
        return self.total_samples

    def _find_shard(self, index: int) -> Tuple[LMDBShardInfo, int]:
        # index: 0-based global index
        # prefix-sum 기반 빠른 검색 (선형으로도 충분하지만 조금 최적화)
        lo, hi = 0, len(self.shards) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s = self.shards[mid]
            if index < s.offset:
                hi = mid - 1
            elif index >= s.offset + s.num_samples:
                lo = mid + 1
            else:
                local = index - s.offset
                return s, local
        raise IndexError(f"global index out of range: {index}")

    def _open_env(self, shard: LMDBShardInfo) -> lmdb.Environment:
        if shard.env is None:
            shard.env = lmdb.open(
                shard.shard_path,
                readonly=True,
                lock=False,
                readahead=True,
                subdir=True,
                max_readers=64,
            )
        return shard.env

    def __getitem__(self, idx: int):
        shard, local_idx = self._find_shard(idx)
        env = self._open_env(shard)

        key = f"{local_idx:012d}".encode("ascii")

        with env.begin(write=False) as txn:
            buf = txn.get(key)
            if buf is None:
                raise KeyError(f"Key not found: shard={shard.shard_index}, key={key!r}")
            obj = msgpack.unpackb(buf, raw=False)

        tensor_bytes = obj["tensor"]
        shape = obj["shape"]  # [18,8,8]
        dtype = obj["dtype"]
        label = obj["label"]
        policy = int(label["policy"])
        value = float(label["value"])

        arr = np.frombuffer(tensor_bytes, dtype=np.uint8)
        arr = arr.reshape(shape)  # (18,8,8)

        # float32, 0~1 스케일 (LMDB는 0/1 uint8)
        x = torch.from_numpy(arr).float() / 255.0

        policy_t = torch.tensor(policy, dtype=torch.long)
        value_t = torch.tensor(value, dtype=torch.float32)

        return x, policy_t, value_t


# ---------------------------
# 모델 정의
# ---------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out, inplace=True)
        return out


class ChessNet(nn.Module):
    def __init__(self, channels: int = 64, num_blocks: int = 4, policy_size: int = 4096):
        super().__init__()
        self.stem = nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(channels)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels))
        self.blocks = nn.Sequential(*blocks)

        # 8x8 board → 64 spatial positions
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, policy_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: [B,18,8,8], float32, 0~1
        x = self.stem(x)
        x = self.bn_stem(x)
        x = F.relu(x, inplace=True)
        x = self.blocks(x)

        policy_logits = self.policy_head(x)  # [B,4096]
        value = self.value_head(x)           # [B,1], -1~1

        return policy_logits, value


# ---------------------------
# 학습 루프
# ---------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    total_epochs,
    log_interval=100,
    writer: SummaryWriter | None = None,   # ★ 수정 포인트
    value_loss_weight: float = 1.0,
):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch")
    global_step = epoch * len(loader)

    avg_loss = 0.0
    avg_policy_loss = 0.0
    avg_value_loss = 0.0
    count = 0

    for i, (x, policy_t, value_t) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        policy_t = policy_t.to(device, non_blocking=True)
        value_t = value_t.to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            policy_logits, value_pred = model(x)
            policy_loss = ce_loss_fn(policy_logits, policy_t)
            value_loss = mse_loss_fn(value_pred, value_t)
            loss = policy_loss + value_loss_weight * value_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        count += 1
        avg_loss += loss.item()
        avg_policy_loss += policy_loss.item()
        avg_value_loss += value_loss.item()

        if i % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{avg_loss / max(1, count):.4f}",
                "p_loss": f"{avg_policy_loss / max(1, count):.4f}",
                "v_loss": f"{avg_value_loss / max(1, count):.4f}",
            })

        if writer is not None:
            step = global_step + i
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/policy_loss", policy_loss.item(), step)
            writer.add_scalar("train/value_loss", value_loss.item(), step)

    avg_loss /= max(1, count)
    avg_policy_loss /= max(1, count)
    avg_value_loss /= max(1, count)

    return avg_loss, avg_policy_loss, avg_value_loss


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    save_dir: str,
    filename: str = "chessnet_last.pt",
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )
    print(f"[CKPT] saved: {path}")


def export_torchscript(model, device, save_dir: str, filename: str = "chessnet_ts.pt"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    model.eval()
    model.to(device)

    # trace 대신 script 사용
    scripted = torch.jit.script(model)
    scripted.save(path)

    print(f"[TS] exported TorchScript: {path}")


# ---------------------------
# main
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb_root", type=str, default=DEFAULT_LMDB_ROOT)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=0)  # LMDB 안전 위해 기본 0
    p.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    p.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR)
    p.add_argument("--value_loss_weight", type=float, default=1.0)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--export_ts", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("=== train_from_lmdb_final ===")
    print(f"lmdb_root: {args.lmdb_root}")
    print(f"device   : {args.device}")

    device = torch.device(args.device)

    dataset = LMDBDataset(args.lmdb_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = ChessNet()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0

    if args.resume:
        ckpt_path = args.resume
        if os.path.isfile(ckpt_path):
            print(f"[CKPT] loading: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0) + 1
        else:
            print(f"[CKPT] resume file not found: {ckpt_path}")

    writer = SummaryWriter(log_dir=args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        avg_loss, avg_policy_loss, avg_value_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device,
            epoch,
            args.epochs,
            writer=writer,
            value_loss_weight=args.value_loss_weight,
        )

        print(
            f"[Epoch {epoch}] loss={avg_loss:.4f}, "
            f"p_loss={avg_policy_loss:.4f}, v_loss={avg_value_loss:.4f}"
        )

        save_checkpoint(
            model,
            optimizer,
            scaler,
            epoch,
            args.save_dir,
            filename=f"chessnet_epoch_{epoch}.pt",
        )
        save_checkpoint(
            model,
            optimizer,
            scaler,
            epoch,
            args.save_dir,
            filename="chessnet_last.pt",
        )

    writer.close()

    if args.export_ts:
        export_torchscript(model, device, args.save_dir)

    print("=== training finished ===")


if __name__ == "__main__":
    main()
