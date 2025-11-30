import torch
import torch.nn as nn

class DummyChessNet(nn.Module):
    def forward(self, x):
        policy = torch.zeros(4096)
        value = torch.tensor([0.0])
        return policy, value

model = DummyChessNet().eval()
example = torch.zeros(1,18,8,8)

scripted = torch.jit.script(model)
scripted.save("dummy_safe.pt")

print("saved dummy_safe.pt")
