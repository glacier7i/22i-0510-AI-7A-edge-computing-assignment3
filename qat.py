"""Q1.3 - Quantization-Aware Training: init from FP32 -> fake-quant -> fine-tune (GPU) -> convert to INT8 (CPU)."""
import argparse

import torch
from torch.ao.quantization import convert, disable_observer, get_default_qat_qconfig, prepare_qat
from torch.nn.intrinsic.qat import freeze_bn_stats

from common import accuracy, fit, get_device, load_cifar, load_model, pick_backend, save_int8, save_log, seed_all, split_val

BACKEND = pick_backend()


def prepare_model():
    """Load the pretrained FP32 weights, fuse, and insert fake-quantization modules."""
    model = load_model("fp32", "cpu").train()               # QAT fusion requires train mode
    model.fuse_model(is_qat=True)
    torch.backends.quantized.engine = BACKEND
    model.qconfig = get_default_qat_qconfig(BACKEND)
    return prepare_qat(model, inplace=True)


def finetune(model, data, epochs, lr, history):
    """QAT fine-tuning on the GPU. Observers settle first, then ranges and BN statistics are frozen so the last
    epochs train against fixed quantization parameters."""
    xtr, ytr, xval, yval = data

    def schedule(ep):
        if ep == epochs * 5 // 8:
            model.apply(disable_observer)
        if ep == epochs * 3 // 4:
            model.apply(freeze_bn_stats)

    return fit(model, xtr, ytr, epochs, lr, eval_data=(xval, yval), amp=False, on_epoch=schedule, tag="qat", history=history)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.005)
    args = ap.parse_args()

    seed_all(0)
    dev = get_device()
    xtr, ytr, xte, yte = load_cifar()
    data = tuple(t.to(dev) for t in split_val(xtr, ytr))

    history, model = [], prepare_model().to(dev)
    finetune(model, data, args.epochs, args.lr, history)
    save_log(history, "qat")

    model = convert(model.cpu().eval(), inplace=False)      # fake-quant -> real INT8 kernels
    print(f"\nQAT INT8 accuracy: {accuracy(model, xte, yte):.2f}%")
    save_int8(model, "qat_int8")
    print("saved checkpoints/qat_int8.pt")
