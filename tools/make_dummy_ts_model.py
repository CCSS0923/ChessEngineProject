import torch
import torch.nn as nn

# 더미 모델 (구조만 맞으면 됨)
class DummyChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(18*8*8, 4096)
        self.value = nn.Linear(18*8*8, 1)

    def forward(self, x):
        # x shape: [B,18,8,8]
        b = x.size(0)
        flat = x.view(b, -1).float()
        return self.policy(flat), torch.tanh(self.value(flat))

model = DummyChessNet().eval()

example = torch.zeros(1,18,8,8)
scripted = torch.jit.script(model)

scripted.save("../checkpoints/dummy_chessnet_ts.pt")
print("saved dummy model")
