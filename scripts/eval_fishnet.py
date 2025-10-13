# scripts/eval_fishnet.py  (REPLACE WHOLE FILE)
import argparse, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader
from fishnet.model import build_model
from fishnet.data.deepfish import CocoDetection
from fishnet.utils.post import decode_and_nms, to_coco_detection_json, save_json, coco_evaluate

def collate_fn(batch):
    imgs, targets, metas = [], [], []
    for im, t, m in batch: imgs.append(im); targets.append(t); metas.append(m)
    import torch as T; return T.stack(imgs,0), targets, metas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--mps", action="store_true")
    ap.add_argument("--out", default="runs/eval/coco_pred.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    val_json = cfg.get("test_json", cfg.get("val_json"))
    images_dir = cfg["images_dir"]
    num_classes = int(cfg.get("num_classes", 1))

    device = torch.device("mps" if args.mps and torch.backends.mps.is_available() else "cpu")

    ds = CocoDetection(val_json, images_dir, img_size=args.img_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = build_model(num_classes=num_classes)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False)
    model.eval().to(device)

    all_dets, all_ids, all_scales = [], [], []
    with torch.no_grad():
        for imgs, targets, metas in dl:
            imgs = imgs.to(device)
            head = model.forward_heads(imgs)
            dets = decode_and_nms(head, img_size=(args.img_size, args.img_size),
                                  conf_thres=args.conf, iou_thres=args.iou,
                                  num_classes=num_classes,
                                  box_format=head.get("box_format","xywh"))
            for det, m in zip(dets, metas):
                all_dets.append(det.cpu())
                all_ids.append(int(m["image_id"]))
                all_scales.append((float(m["scale_w"]), float(m["scale_h"])))

    preds = to_coco_detection_json(all_dets, all_ids, all_scales, class_offset=1)
    out_path = Path(args.out); save_json(preds, out_path)
    stats = coco_evaluate(Path(val_json), out_path, iou_type="bbox")
    print("\n== COCO mAP =="); [print(f"{k}: {v:.4f}") for k,v in stats.items()]

if __name__ == "__main__":
    main()
