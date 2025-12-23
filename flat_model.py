import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class FlatRefiner(nn.Module):
    def __init__(self, in_channels=5, channels=64, num_blocks=8):
        super().__init__()

        # 5 → 64
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # many residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(channels))
        self.body = nn.Sequential(*blocks)

        # 64 → 1 (zero init)
        self.tail = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        residual = self.tail(x)
        return residual

