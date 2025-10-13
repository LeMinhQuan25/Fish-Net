# tests/sanity_check.py  (REPLACE WHOLE FILE)
import torch
from fishnet.model import build_model

m = build_model(num_classes=1)
m.eval()
x = torch.randn(1,3,640,640)
with torch.no_grad():
    out = m.forward_heads(x)
print("OK shapes:", out["boxes"].shape, out["scores"].shape, "format:", out["box_format"])
