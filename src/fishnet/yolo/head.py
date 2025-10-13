# DetectHead: nhánh cls/box/obj cho mỗi cấp đặc trưng
from __future__ import annotations
import torch
import torch.nn as nn
from .blocks import Conv

class DetectHead(nn.Module):
    def __init__(self, ch, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = nn.ModuleList([Conv(c, c) for c in ch])
        self.box_convs = nn.ModuleList([Conv(c, c) for c in ch])
        self.cls_pred = nn.ModuleList([nn.Conv2d(c, num_classes, 1) for c in ch])
        self.box_pred = nn.ModuleList([nn.Conv2d(c, 4, 1) for c in ch])
        self.obj_pred = nn.ModuleList([nn.Conv2d(c, 1, 1) for c in ch])

    def forward(self, feats):
        boxes, logits, obj = [], [], []
        for i, f in enumerate(feats):
            c = self.cls_convs[i](f)
            b = self.box_convs[i](f)
            logits.append(self.cls_pred[i](c))  # (B,C,H,W)
            boxes.append(self.box_pred[i](b))   # (B,4,H,W)
            obj.append(self.obj_pred[i](b))     # (B,1,H,W)
        return boxes, logits, obj
