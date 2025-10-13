# CharbonnierLoss; FocalBCE; CIoU và DetectionLoss dạng YOLO-ish
import math, torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__(); self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))

def bbox_iou_ciou(box1, box2, eps=1e-7):
    (x1,y1,x2,y2) = box1.unbind(-1); (x1g,y1g,x2g,y2g) = box2.unbind(-1)
    w1 = (x2-x1).clamp(min=0); h1 = (y2-y1).clamp(min=0)
    w2 = (x2g-x1g).clamp(min=0); h2 = (y2g-y1g).clamp(min=0)
    inter_x1 = torch.max(x1, x1g); inter_y1 = torch.max(y1, y1g)
    inter_x2 = torch.min(x2, x2g); inter_y2 = torch.min(y2, y2g)
    inter = (inter_x2-inter_x1).clamp(min=0) * (inter_y2-inter_y1).clamp(min=0)
    union = w1*h1 + w2*h2 - inter + eps
    iou = inter/union
    cx1, cy1 = (x1+x2)/2, (y1+y2)/2; cx2, cy2 = (x1g+x2g)/2, (y1g+y2g)/2
    rho2 = (cx2-cx1)**2 + (cy2-cy1)**2
    c_x1 = torch.min(x1, x1g); c_y1 = torch.min(y1, y1g)
    c_x2 = torch.max(x2, x2g); c_y2 = torch.max(y2, y2g)
    c2 = (c_x2-c_x1)**2 + (c_y2-c_y1)**2 + eps
    v = (4/(math.pi**2)) * torch.pow(torch.atan(w2/(h2+eps)) - torch.atan(w1/(h1+eps)), 2)
    with torch.no_grad(): alpha = v / (1 - iou + v + eps)
    return iou - (rho2/c2 + alpha*v)

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.reduction=reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets):
        bce = self.bce(logits, targets); p = torch.sigmoid(logits)
        pt = targets*p + (1-targets)*(1-p); w = self.alpha * (1-pt)**self.gamma
        loss = w * bce
        if self.reduction=="mean": return loss.mean()
        if self.reduction=="sum": return loss.sum()
        return loss

class DetectionLossYOLOish(nn.Module):
    def __init__(self, box_lambda=5.0, obj_lambda=1.0, cls_lambda=1.0):
        super().__init__(); self.box_lambda=box_lambda; self.obj_lambda=obj_lambda; self.cls_lambda=cls_lambda
        self.focal = FocalBCEWithLogitsLoss(0.25, 2.0)
    def forward(self, pred, target):
        ciou = bbox_iou_ciou(pred["pred_boxes"], target["tgt_boxes"]).clamp(min=-1,max=1)
        box_loss = (1.0 - ciou).mean()
        obj_loss = self.focal(pred["pred_obj"], target["tgt_obj"]).mean()
        cls_loss = self.focal(pred["pred_cls"], target["tgt_cls"]).mean()
        total = self.box_lambda*box_loss + self.obj_lambda*obj_loss + self.cls_lambda*cls_loss
        return {"total": total, "box": box_loss, "obj": obj_loss, "cls": cls_loss}
