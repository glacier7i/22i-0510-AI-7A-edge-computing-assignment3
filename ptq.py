"""Q1.2 - Post-Training Static Quantization: fuse -> configure backend -> calibrate -> convert -> save."""
import argparse

import torch
from torch.ao.quantization import convert, get_default_qconfig, prepare

from common import accuracy, load_cifar, load_model, pick_backend, save_int8, seed_all, split_val

BACKEND = pick_backend()


@torch.no_grad()
def calibrate(model, x, bs=100):
    """Calibration pipeline: forward a training subset through the observed model so every observer records the
    activation range used to compute the INT8 scale / zero-point."""
    model.eval()
    for i in range(0, len(x), bs):
        model(x[i:i + bs])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-images", type=int, default=2000, help="size of the training subset used for calibration")
    args = ap.parse_args()

    seed_all(0)
    xtr, ytr, xte, yte = load_cifar()
    xtr, ytr, _, _ = split_val(xtr, ytr)                   # same 45k training split as every other script

    model = load_model("fp32", "cpu")
    print(f"FP32 accuracy: {accuracy(model, xte, yte):.2f}%")

    model.fuse_model(is_qat=False)                         # 1) fuse Conv+BN(+ReLU)
    torch.backends.quantized.engine = BACKEND              # 2) quantization backend
    model.qconfig = get_default_qconfig(BACKEND)
    prepare(model, inplace=True)                           # insert observers

    calibrate(model, xtr[torch.randperm(len(xtr))[:args.calib_images]])  # 3) calibration on a training subset
    convert(model, inplace=True)                           # 4) FP32 -> INT8
    print(f"PTQ INT8 accuracy: {accuracy(model, xte, yte):.2f}%")
    save_int8(model, "ptq_int8")                           # 5) save for deployment
    print("saved checkpoints/ptq_int8.pt")
