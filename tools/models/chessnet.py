import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """
    입력: [B, 18, 8, 8]
    출력:
      - policy: [B, 4096]  (from*64 + to 인덱스)
      - value : [B, 1]     (-1 ~ 1 스칼라)
    """

    def __init__(self, channels: int = 128, num_blocks: int = 6):
        super().__init__()
        self.input_conv = nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels))
        self.res_blocks = nn.Sequential(*blocks)

        # policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

        # value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # shared trunk
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out)

        out = self.res_blocks(out)

        # policy head
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)        # [B, 32*8*8]
        p = self.policy_fc(p)           # [B, 4096]

        # value head
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)       # [B, 32*8*8]
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_fc2(v)           # [B, 1]
        v = torch.tanh(v)               # [-1, 1]

        return p, v
