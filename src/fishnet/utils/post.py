# decode+NMS; lưu JSON; COCO eval (pycocotools)
from pathlib import Path
from typing import List, Tuple, Dict
import json, cv2, numpy as np, torch
from torchvision.ops import nms

def xywh2xyxy(xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = xywh.unbind(-1)
    return torch.stack((x - w/2, y - h/2, x + w/2, y + h/2), dim=-1)

def clip_boxes(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    boxes[..., 0::2].clamp_(0, w-1); boxes[..., 1::2].clamp_(0, h-1); return boxes

def decode_and_nms(head_outs: Dict[str, torch.Tensor], img_size: Tuple[int,int],
                   conf_thres=0.25, iou_thres=0.5, max_det=300,
                   num_classes=1, box_format="xywh") -> List[torch.Tensor]:
    boxes = head_outs["boxes"]; scores = head_outs["scores"].sigmoid()  # logits→probs
    if box_format == "xywh": boxes = xywh2xyxy(boxes)
    H, W = img_size; boxes = clip_boxes(boxes, H, W)
    out = []
    for b in range(boxes.shape[0]):
        bboxes = boxes[b]; prob, cls = scores[b].max(dim=-1)
        keep = prob > conf_thres
        if keep.sum() == 0:
            out.append(torch.zeros((0,6), device=boxes.device)); continue
        bboxes, prob, cls = bboxes[keep], prob[keep], cls[keep].float()
        keep_idx = nms(bboxes, prob, iou_thres)[:max_det]
        det = torch.cat([bboxes[keep_idx], prob[keep_idx,None], cls[keep_idx,None]], dim=1)
        out.append(det)
    return out

def draw_boxes(img_bgr: np.ndarray, det: torch.Tensor, class_names: List[str]) -> np.ndarray:
    img = img_bgr.copy()
    if det.numel()==0: return img
    for i in range(det.shape[0]):
        x1,y1,x2,y2,s,c = det[i].tolist()
        x1,y1,x2,y2 = map(int,[x1,y1,x2,y2]); name = class_names[int(c)] if int(c)<len(class_names) else str(int(c))
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        label = f"{name} {s:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img,(x1,y1-th-6),(x1+tw+4,y1),(0,255,0),-1)
        cv2.putText(img,label,(x1+2,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    return img

def to_coco_detection_json(all_dets: List[torch.Tensor], image_ids: List[int],
                           scale_factors: List[Tuple[float,float]], class_offset=1) -> List[Dict]:
    res = []
    for det, img_id, (wx,hy) in zip(all_dets, image_ids, scale_factors):
        if det.numel()==0: continue
        for x1,y1,x2,y2,score,cls in det.cpu().numpy():
            x1o,y1o,x2o,y2o = x1/wx, y1/hy, x2/wx, y2/hy
            res.append(dict(image_id=int(img_id),
                            category_id=int(cls)+class_offset,
                            bbox=[float(x1o),float(y1o),float(x2o-x1o),float(y2o-y1o)],
                            score=float(score)))
    return res

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")

def coco_evaluate(gt_json_path: Path, det_json_path: Path, iou_type="bbox"):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(str(gt_json_path)); coco_dt = coco_gt.loadRes(str(det_json_path))
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP@[.5:.95]": float(ev.stats[0]), "AP@0.50": float(ev.stats[1]),
            "AP@0.75": float(ev.stats[2]), "AP_small": float(ev.stats[3]),
            "AP_medium": float(ev.stats[4]), "AP_large": float(ev.stats[5]),
            "AR@[.5:.95]": float(ev.stats[6])}
