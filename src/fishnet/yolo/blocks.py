# Các block Conv, C2f, SCDown, C2fCIB (YOLOv10-like)
from __future__ import annotations
import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1, 0)
        self.m = nn.Sequential(*[Conv(c2, c2, 3) for _ in range(n)])
    def forward(self, x):
        x = self.cv1(x)
        return self.m(x)

class SCDown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv = Conv(c1, c2, 3, 2)
    def forward(self, x):
        return self.cv(x)

class C2fCIB(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1, 0)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2, 1),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 1),
            nn.Sigmoid(),
        )
        self.m = nn.Sequential(*[Conv(c2, c2, 3) for _ in range(n)])
    def forward(self, x):
        x = self.cv1(x)
        a = self.attn(x)
        x = x * a
        return self.m(x)
