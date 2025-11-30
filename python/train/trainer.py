import torch
from torch.optim import Adam
from train.losses import loss_policy, loss_value

class Trainer:
    def __init__(self, model, lr=1e-3, device="cuda"):
        self.model = model.to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for x, p, v in loader:
            x = x.to(self.device)
            p = p.to(self.device)
            v = v.to(self.device)
            policy_pred, value_pred = self.model(x)
            lp = loss_policy(policy_pred, p)
            lv = loss_value(value_pred, v)
            loss = lp + lv
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
