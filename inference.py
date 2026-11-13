"""Inference pipeline for all five models: fp32 | ptq_int8 | qat_int8 | unstructured | structured.

  python inference.py --model qat_int8 --n 8                  # first 8 CIFAR-10 test images
  python inference.py --model structured --images a.png b.jpg # your own images (resized to 32x32)
"""
import argparse

import numpy as np
import torch
from PIL import Image

from common import CKPTS, CLASSES, load_cifar, load_model, normalize


def load_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32))
    return normalize(torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255)[None])


@torch.no_grad()
def classify(model, x, device="cpu"):
    """x: normalized (N,3,32,32). Returns (class indices, confidences)."""
    probs = model(x.to(device)).softmax(1).cpu()
    conf, idx = probs.max(1)
    return idx, conf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=CKPTS, default="fp32")
    ap.add_argument("--images", nargs="*", help="image files; default: CIFAR-10 test images")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (INT8 models are CPU-only)")
    args = ap.parse_args()

    device = "cpu" if args.model.endswith("_int8") else args.device
    model = load_model(args.model, device)
    if args.images:
        x, truth, names = torch.cat([load_image(p) for p in args.images]), None, args.images
    else:
        _, _, xte, yte = load_cifar()
        x, truth, names = xte[:args.n], yte[:args.n], [f"test[{i}]" for i in range(args.n)]

    idx, conf = classify(model, x, device)
    print(f"model: {args.model} on {device}")
    for i, name in enumerate(names):
        line = f"{name:>12}: {CLASSES[idx[i]]:<10} ({conf[i]:.1%})"
        if truth is not None:
            line += f"  true={CLASSES[truth[i]]:<10} {'OK' if idx[i] == truth[i] else 'WRONG'}"
        print(line)
