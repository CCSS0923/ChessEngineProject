import torch
from torch.utils.data import DataLoader
from models.chessnet import ChessNet
from data.dataset_lmdb import ChessLMDBDataset
from train.trainer import Trainer
from train.config import *

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ChessLMDBDataset(DATA_PATH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, pin_memory=True)
    model = ChessNet()
    trainer = Trainer(model, lr=LR, device=device)
    for epoch in range(1, EPOCHS+1):
        loss = trainer.train_epoch(loader)
        print(f"[Epoch {epoch}] loss={loss:.6f}")
        torch.jit.script(model).save(f"checkpoints/chessnet_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()
