import torch
import torch.nn as nn

from attrackt.autoencoder.utils import nonlinearity, normalize


class ResnetBlock(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        num_out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.norm1 = normalize(in_channels=num_in_channels)
        self.conv1 = torch.nn.Conv2d(
            num_in_channels, num_out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = normalize(num_out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            num_out_channels, num_out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.num_in_channels != self.num_out_channels:
            self.nin_shortcut = torch.nn.Conv2d(
                num_in_channels,
                num_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.num_in_channels != self.num_out_channels:
            x = self.nin_shortcut(x)

        return x + h
