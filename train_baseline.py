"""Q1.1 - Train the FP32 ResNet-18 baseline on CIFAR-10 and save the checkpoint."""
import argparse

import torch

from common import accuracy, build_model, fit, get_device, load_cifar, save_fp, save_log, seed_all, split_val

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    seed_all(0)
    dev = get_device()
    xtr, ytr, xte, yte = load_cifar()
    xtr, ytr, xval, yval = (t.to(dev) for t in split_val(xtr, ytr))  # 45k train / 5k validation
    xte, yte = xte.to(dev), yte.to(dev)

    history = []
    model = build_model().to(dev).to(memory_format=torch.channels_last)
    fit(model, xtr, ytr, args.epochs, args.lr, eval_data=(xval, yval), smoothing=0.1, tag="fp32", history=history)

    save_log(history, "fp32")
    save_fp(model.cpu(), "fp32")
    print(f"\nFP32 baseline test accuracy: {accuracy(model.to(dev), xte, yte, dev):.2f}%  -> checkpoints/fp32.pth")
