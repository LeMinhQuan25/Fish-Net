# Định nghĩa FishNet: FI → Backbone → Neck → Head (+ tiện ích flatten)
from __future__ import annotations
import torch
import torch.nn as nn
from .modules.fi import FIModule
from .yolo.backbone import Backbone
from .yolo.neck import Neck
from .yolo.head import DetectHead

class FishNet(nn.Module):
    def __init__(self, num_classes=1, fi_channels=64, fi_blocks=4, swin_heads=8, swin_window=7, swin_mlp_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.fi = FIModule(3, fi_channels, fi_blocks, swin_heads, swin_window, swin_mlp_dim)
        self.backbone = Backbone(in_ch=fi_channels)
        self.neck = Neck()

        # DetectHead sẽ được áp dụng trên các feature đã được Neck tinh chỉnh
        # Số kênh sẽ được Neck giữ nguyên từ backbone; Head không cần biết ch ở đây
        # Ta sẽ suy từ tensor thực tế ở runtime
        self.head = None  # sẽ khởi tạo lười (lazy) ở lần forward đầu

    def _lazy_init_head(self, feats):
        # feats: [p3, p4, p5]
        c3, c4, c5 = [f.shape[1] for f in feats]
        from .yolo.head import DetectHead
        self.head = DetectHead([c3, c4, c5], num_classes=self.num_classes)
        self.add_module("detect_head", self.head)
        # 🔧 move head to the same device as feats
        self.head.to(feats[0].device)

    def forward(self, x):
        f = self.fi(x)
        feats = self.backbone(f)       # [p3, p4, p5]
        neck = self.neck(feats)        # [p3', p4', p5']

        if self.head is None:
            self._lazy_init_head(neck)

        boxes, logits, obj = self.head(neck)  # lists of (B,*,H,W)
        return {"raw_boxes": boxes, "raw_logits": logits, "raw_obj": obj}

    @staticmethod
    def _flatten_levels(level_tensors):
        outs = []
        for t in level_tensors:
            B, C, H, W = t.shape
            outs.append(t.permute(0, 2, 3, 1).contiguous().view(B, H * W, C))
        return torch.cat(outs, dim=1)  # (B, sum(HW), C)

    def forward_heads(self, x):
        out = self.forward(x)
        boxes = self._flatten_levels(out["raw_boxes"])   # (B,N,4)  (giả định head đang xuất xywh center-based)
        logits = self._flatten_levels(out["raw_logits"]) # (B,N,C)
        obj = self._flatten_levels(out["raw_obj"])       # (B,N,1)
        scores = logits  # để decoder sigmoid
        return {"boxes": boxes, "scores": scores, "obj": obj, "box_format": "xywh"}

def build_model(num_classes=1, **kwargs):
    return FishNet(num_classes=num_classes, **kwargs)
