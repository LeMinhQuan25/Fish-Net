# scripts/train_fishnet.py  (REPLACE WHOLE FILE)
import argparse, yaml, torch, os, psutil
from torch.utils.data import DataLoader
from torch.optim import AdamW
from fishnet.model import build_model
from fishnet.data.deepfish import CocoDetection

def collate_fn(batch):
    imgs, targets, metas = [], [], []
    for im, t, m in batch: imgs.append(im); targets.append(t); metas.append(m)
    import torch as T; return T.stack(imgs,0), targets, metas

def mem():
    # tiện in RAM hệ thống để bạn theo dõi
    m = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    return f"{m:.2f} GB RSS"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--mps', action='store_true')
    ap.add_argument('--amp', action='store_true', help='use mixed precision on MPS')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    use_mps = args.mps and torch.backends.mps.is_available() and cfg.get('use_mps', True)
    device = torch.device('mps' if use_mps else 'cpu')
    print(f"[device] {device} | AMP={args.amp} | mem={mem()}")

    train_ds = CocoDetection(cfg['train_json'], cfg['images_dir'], cfg['img_size'])
    val_ds   = CocoDetection(cfg['val_json'],   cfg['images_dir'], cfg['img_size'])
    train_ld = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=0, collate_fn=collate_fn)
    val_ld   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = build_model(num_classes=int(cfg.get('num_classes',1)))
    model.to(device)
    opt = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    autocast_ctx = (torch.autocast(device_type="mps", dtype=torch.float16) if (device.type=="mps" and args.amp)
                    else torch.autocast(device_type="cpu", enabled=False))

    for epoch in range(args.epochs):
        model.train()
        for i, (imgs, targets, metas) in enumerate(train_ld):
            imgs = imgs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx:
                head = model.forward_heads(imgs)
                # DUMMY LOSS có grad: dùng trực tiếp logits
                loss = head['scores'].mean()          # <—— THAY cho torch.zeros(...)
            loss.backward()
            opt.step()

            if i % 5 == 0:
                print(f"[Epoch {epoch}] step {i}  boxes={head['boxes'].shape} scores={head['scores'].shape}  | mem={mem()}")

        model.eval()
        with torch.no_grad():
            for imgs, targets, metas in val_ld:
                imgs = imgs.to(device, non_blocking=True)
                with autocast_ctx:
                    _ = model.forward_heads(imgs)
                break

    os.makedirs("runs/fishnet", exist_ok=True)
    torch.save({"model": model.state_dict()}, "runs/fishnet/best.pt")
    print("Saved: runs/fishnet/best.pt")

if __name__ == "__main__":
    main()
