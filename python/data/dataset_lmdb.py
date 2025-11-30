import lmdb
import msgpack
import torch
from torch.utils.data import Dataset

class ChessLMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, max_readers=64)
        with self.env.begin() as txn:
            self.length = txn.stat()["entries"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = f"{idx:08d}".encode()
        with self.env.begin() as txn:
            data = txn.get(key)
            if data is None:
                raise IndexError(idx)
        sample = msgpack.unpackb(data, raw=False)
        x = torch.tensor(sample["x"], dtype=torch.float32) / 255.0
        p = sample["policy"]
        v = float(sample["value"])
        return x, p, v
