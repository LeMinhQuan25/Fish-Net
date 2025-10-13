import os, yaml, argparse, torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from fishnet.modules.fi import FIModule
from fishnet.losses import CharbonnierLoss

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config))

device = 'mps' if torch.backends.mps.is_available() and cfg.get('use_mps', True) else 'cpu'

# GIẢ ĐỊNH LOL_DIR có cấu trúc pairs low/gt theo thư mục ImageFolder
lol_root = os.environ.get('LOL_DIR', '')
low_dir = os.path.join(lol_root, 'low')
gt_dir  = os.path.join(lol_root, 'gt')

transform = transforms.Compose([
    transforms.Resize((cfg['img_size'], cfg['img_size'])),
    transforms.ToTensor(),
])
low = ImageFolder(low_dir, transform=transform)
gt  = ImageFolder(gt_dir,  transform=transform)
assert len(low)==len(gt), 'LOL low/gt phải cùng số lượng'

loader = DataLoader(list(zip(low, gt)), batch_size=4, shuffle=True)

model = FIModule(3, cfg['fi_channels'], cfg['fi_blocks'], cfg['swin_heads'], cfg['swin_window'], cfg['swin_mlp_dim']).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = CharbonnierLoss()

# Head tái dựng proxy
recon_head = torch.nn.Conv2d(cfg['fi_channels'], 3, 1).to(device)

model.train()
for e in range(args.epochs):
    total = 0.0
    for (xl, _), (yg, _) in loader:
        xl, yg = xl.to(device), yg.to(device)
        yhat = model(xl)  # đặc trưng
        recon = recon_head(yhat)
        loss = crit(recon, yg)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()*xl.size(0)
    print(f"[FI] epoch {e+1}: {total/len(loader.dataset):.4f}")

os.makedirs('runs/fi', exist_ok=True)
torch.save(model.state_dict(), 'runs/fi/fi_pretrained.pt')
print('Saved runs/fi/fi_pretrained.pt')
