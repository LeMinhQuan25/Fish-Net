# Backbone: stem → stage2(C2f+SCDown) → stage3(C2f+SCDown) → stage4(C2fCIB)
from __future__ import annotations
import torch.nn as nn
from .blocks import Conv, C2f, SCDown, C2fCIB

class Backbone(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        c2, c3, c4 = 128, 256, 512
        self.stem = Conv(in_ch, c2, 3, 1)
        self.stage2 = nn.Sequential(C2f(c2, c2), SCDown(c2, c3))
        self.stage3 = nn.Sequential(C2f(c3, c3), SCDown(c3, c4))
        self.stage4 = C2fCIB(c4, c4)
        self.out_channels = (c2, c3, c4)
    def forward(self, x):
        p2 = self.stem(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]
