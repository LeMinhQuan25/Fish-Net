# Neck FPN-like, tạo conv “lateral/out” động theo kênh đầu vào
from __future__ import annotations
import torch
import torch.nn as nn
from .blocks import Conv

class Neck(nn.Module):
    """
    FPN-like neck with dynamic channel init on first forward.
    It adapts to the actual channels of (p3, p4, p5) coming from the backbone,
    so we never hit a mismatch like 256 vs 512 when adding skip connections.
    """
    def __init__(self):
        super().__init__()
        # Will be created on first forward based on incoming feature channels:
        self.lateral4 = None  # Conv(c5 -> c4)
        self.lateral3 = None  # Conv(c4 -> c3)
        self.out3 = None      # Conv(c3 -> c3)
        self.out4 = None      # Conv(c4 -> c4)
        self.out5 = None      # Conv(c5 -> c5)

    # src/fishnet/yolo/neck.py  (PATCH ONLY _build)
    def _build(self, feats):
        """Build convs with the correct in/out channels using the real shapes."""
        p3, p4, p5 = feats
        c3, c4, c5 = p3.shape[1], p4.shape[1], p5.shape[1]

        # 1x1 lateral to match channels for addition
        self.lateral4 = Conv(c5, c4, k=1, s=1, p=0)
        self.lateral3 = Conv(c4, c3, k=1, s=1, p=0)

        # small 3x3 convs to smooth outputs
        self.out3 = Conv(c3, c3)
        self.out4 = Conv(c4, c4)
        self.out5 = Conv(c5, c5)

        # register as modules
        self.add_module("lateral4", self.lateral4)
        self.add_module("lateral3", self.lateral3)
        self.add_module("out3", self.out3)
        self.add_module("out4", self.out4)
        self.add_module("out5", self.out5)

        # 🔧 move newly-created submodules to the same device as inputs
        device = p3.device
        self.lateral4.to(device)
        self.lateral3.to(device)
        self.out3.to(device)
        self.out4.to(device)
        self.out5.to(device)

    def forward(self, feats):
        # feats should be [p3, p4, p5] from backbone
        p3, p4, p5 = feats

        if self.lateral4 is None:
            # First call: infer channels and create layers
            self._build(feats)

        # Top-down pathway with proper channel matching
        l4 = self.lateral4(p5)  # (B, c4, H4, W4)
        up4 = torch.nn.functional.interpolate(l4, size=p4.shape[-2:], mode='nearest')
        p4 = self.out4(p4 + up4)

        l3 = self.lateral3(p4)  # (B, c3, H3, W3)
        up3 = torch.nn.functional.interpolate(l3, size=p3.shape[-2:], mode='nearest')
        p3 = self.out3(p3 + up3)

        # (Optional) refine p5 too
        p5 = self.out5(p5)

        return [p3, p4, p5]
