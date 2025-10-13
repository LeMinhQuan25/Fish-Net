.PHONY: setup train eval infer
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt
train:
	python -m pip install -e . && python scripts/train_fishnet.py --config configs/deepfish.yaml --epochs 50
eval:
	python scripts/eval_fishnet.py --config configs/deepfish.yaml --ckpt runs/fishnet/best.pt
infer:
	python scripts/infer_image.py --ckpt runs/fishnet/best.pt --image demo.jpg --out out.jpg
