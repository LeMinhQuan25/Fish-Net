# scripts/infer_image.py  (REPLACE WHOLE FILE)
import argparse, cv2, torch
from pathlib import Path
from fishnet.model import build_model
from fishnet.utils.post import decode_and_nms, draw_boxes

def letterbox(img, new_size=640):
    h0, w0 = img.shape[:2]
    r = min(new_size / h0, new_size / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = (new_size - new_unpad[1]) // 2
    bottom = new_size - new_unpad[1] - top
    left = (new_size - new_unpad[0]) // 2
    right = new_size - new_unpad[0] - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    return img_padded, r, (left, top)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out.jpg")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--classes", default="fish")
    ap.add_argument("--mps", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps" if args.mps and torch.backends.mps.is_available() else "cpu")

    model = build_model(num_classes=len(args.classes.split(",")))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False)
    model.eval().to(device)

    img0 = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img0 is None: raise FileNotFoundError(args.image)
    lb, r, (padw, padh) = letterbox(img0, args.img_size)
    x = torch.from_numpy(cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)).permute(2,0,1).float()[None]/255.0
    x = x.to(device)

    with torch.no_grad():
        head = model.forward_heads(x)
        dets = decode_and_nms(head, img_size=(args.img_size, args.img_size),
                              conf_thres=args.conf, iou_thres=args.iou,
                              num_classes=len(args.classes.split(",")),
                              box_format=head.get("box_format","xywh"))

    det = dets[0].cpu()
    if det.numel():
        det[:, [0,2]] -= padw; det[:, [1,3]] -= padh
        det[:, :4].clamp_(min=0)
        det[:, [0,2]] /= r; det[:, [1,3]] /= r

    out_img = draw_boxes(img0, det, [s.strip() for s in args.classes.split(",")])
    cv2.imwrite(args.out, out_img)
    print(f"Saved: {args.out} — {det.shape[0]} boxes")

if __name__ == "__main__":
    main()
