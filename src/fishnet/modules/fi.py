# FI module: stem conv → (FE × 4) → conv 1x1
from __future__ import annotations
import torch
import torch.nn as nn
from .swin import SwinBlock

class FE(nn.Module):
    """Feature Enhancement: SwinBlock + DeConv"""
    def __init__(self, c, heads=8, window=7, mlp_dim=128):
        super().__init__()
        self.swin = SwinBlock(c, heads, window, mlp_dim)
        # ConvTranspose2d với stride=1, padding=0 sẽ làm tăng kích thước theo H/W
        # sau mỗi FE thêm 1 pixel, phá vỡ tính chia hết cho 2^n cần thiết của backbone.
        # Ta dùng kernel=3, stride=1, padding=1 để giữ nguyên kích thước không gian.
        self.deconv = nn.ConvTranspose2d(c, c, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.swin(x)
        x = self.deconv(x)
        return x

class FIModule(nn.Module):
    def __init__(self, in_ch=3, c=64, blocks=4, heads=8, window=7, mlp_dim=128):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, c, 3, padding=1)
        self.blocks = nn.Sequential(*[FE(c, heads, window, mlp_dim) for _ in range(blocks)])
        self.out_conv1x1 = nn.Conv2d(c, c, 1)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.out_conv1x1(x)
        return x  # đặc trưng đã tăng cường (channels=c)
