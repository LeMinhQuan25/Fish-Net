# CocoDetection: đọc COCO JSON, resize+pad letterbox, trả (img, target, meta)
from pathlib import Path
import json, cv2, torch
from torch.utils.data import Dataset

class CocoDetection(Dataset):
    def __init__(self, json_file, img_dir, img_size=640):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.img_size = int(img_size)
        ann = json.loads(Path(json_file).read_text(encoding="utf-8"))
        self.images = ann["images"]
        self.annotations = ann["annotations"]
        self.ann_by_img = {}
        for a in self.annotations:
            self.ann_by_img.setdefault(a["image_id"], []).append(a)

    def __len__(self): return len(self.images)

    @staticmethod
    def _letterbox(img, new_size=640):
        h0, w0 = img.shape[:2]
        r = min(new_size / h0, new_size / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top = (new_size - new_unpad[1]) // 2
        bottom = new_size - new_unpad[1] - top
        left = (new_size - new_unpad[0]) // 2
        right = new_size - new_unpad[0] - left
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
        # scale factors model→orig
        scale_w = (new_size) / new_unpad[0]
        scale_h = (new_size) / new_unpad[1]
        return img_padded, r, (left, top), scale_w, scale_h

    def __getitem__(self, idx):
        info = self.images[idx]
        img_path = self.img_dir / info["file_name"]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        h0, w0 = img.shape[:2]
        img_lb, r, (padw, padh), scale_w, scale_h = self._letterbox(img, self.img_size)

        anns = self.ann_by_img.get(info["id"], [])
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]  # COCO xywh (orig)
            x1, y1, x2, y2 = x, y, x + w, y + h
            # map to letterboxed (model space)
            x1m = x1 * r + padw; y1m = y1 * r + padh
            x2m = x2 * r + padw; y2m = y2 * r + padh
            boxes.append([x1m, y1m, x2m, y2m])
            labels.append(a.get("category_id", 1) - 1)  # 0-based

        import numpy as np
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img_rgb).permute(2,0,1).float()/255.0
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)
        target = {"boxes": boxes, "labels": labels}
        meta = {
            "image_id": int(info["id"]),
            "scale_w": float(scale_w),
            "scale_h": float(scale_h),
            "padw": int(padw), "padh": int(padh),
            "ratio": float(r), "orig_w": int(w0), "orig_h": int(h0)
        }
        return img_t, target, meta
