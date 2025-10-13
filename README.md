# Fish-Net (FI + YOLOv10-like) cho DeepFish

## Môi trường

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Cấu hình

```bash
cp .env.example .env
# sửa .env cho đường dẫn DATA_DIR, LOL_DIR (nếu pretrain FI)
```

### Giải thích

### modules/fi.py

- FE: SwinBlock → ConvTranspose2d (DeConv).
- FIModule: stem Conv(3→C) → lặp blocks (mặc định 4 FE) → out_conv1x1. Đúng với bài báo 4 FE + Conv 1×1.

### modules/swin.py

- SwinBlock: WindowAttention + MLP + LayerNorm, hoạt động trên tensor B×C×H×W (chuyển vị qua B×H×W×C khi tính attention/MLP).

### losses.py

- CharbonnierLoss(eps=1e-3): đúng loss FI trong paper.
- bbox_iou_ciou + FocalBCEWithLogitsLoss + DetectionLossYOLOish: loss tổng hợp box/obj/cls dạng YOLO.

### yolo/blocks.py

- Định nghĩa Conv, C2f, SCDown, C2fCI

### yolo/backbone.py

- Dùng C2f, SCDown, C2fCIB đúng phong cách YOLOv10-like; trả ra 3 mức đặc trưng [p3, p4, p5].

### yolo/neck.py

- FPN-like: tạo các lateral/out conv động ở lần forward đầu tiên để khớp channel (tránh mismatch kênh). Top-down: p5→p4→p3.

### yolo/head.py

- Cho mỗi mức đặc trưng: cls_convs/box_convs → cls_pred/box_pred/obj_pred ⇒ xuất (B,C,H,W), (B,4,H,W) và (B,1,H,W).

### model.py

- FishNet: ghép FI → Backbone → Neck → DetectHead. Có helper flatten mức (gom (B,4,H,W) → (B, N, 4)…).
- build_model(...) để khởi tạo theo num_classes và các tham số FI (channels, blocks, heads, window, mlp_dim).

### data/deepfish.py

- CocoDetection: đọc images, annotations, group annotation theo image_id; letterbox để khớp img_size; chuẩn hóa toạ độ bbox và labels.

### utils/post.py

- decode_and_nms(...) (sigmoid cls, chuyển xywh→xyxy, clip, NMS), save_json, và coco_evaluate bằng pycocotools.

## Chạy trên Kaggle

### Lệnh train

```bash
%%bash
cd /kaggle/input/fishnet-code
python scripts/train_fishnet.py --config /kaggle/working/deepfish_kaggle.yaml --epochs 10 --amp
```

### Lệnh để tính mAP COCO

```bash
%%bash
cd /kaggle/input/fishnet-code
python scripts/eval_fishnet.py \
  --config /kaggle/working/deepfish_kaggle.yaml \
  --ckpt runs/fishnet/best.pt \
  --batch 4 --conf 0.001 --iou 0.65 \
  --out runs/eval/coco_pred.json
```

### Lệnh suy luận 1 ảnh + vẽ bbox (inference + visualize)

```bash
%%bash
cd /kaggle/input/fishnet-code
python scripts/infer_image.py \
  --ckpt runs/fishnet/best.pt \
  --image /kaggle/input/deepfish-data/deepfish/images/<ten_anh>.jpg \
  --out /kaggle/working/vis.jpg --conf 0.25 --iou 0.5
ls -lh /kaggle/working/vis.jpg
```

## Chạy trên máy local

### Chạy sanity check

```bash
python -m pip install -e .
python tests/sanity_check.py
```

### Tiền huấn luyện FI trên LOL (khuyến nghị)

```bash
python scripts/pretrain_fi.py --config configs/deepfish.yaml --epochs 5
```

### Huấn luyện Fish-Net

```bash
python scripts/train_fishnet.py --config configs/deepfish.yaml --epochs 10
```

### Đánh giá

```bash
python scripts/eval_fishnet.py --config configs/deepfish.yaml --ckpt runs/fishnet/best.pt
```

### Suy luận 1 ảnh

```bash
python scripts/infer_image.py --ckpt runs/fishnet/best.pt --image demo.jpg --out out.jpg
```
