"""Q2.3 / Q2.4 - Structured channel pruning of every BasicBlock's inner channels, physically rebuilt as a smaller net.

For block: x -> conv1(Cin->C) -> bn1 -> relu -> conv2(C->Cout) -> bn2 (+ skip).
Removing channel c of C removes filter c of conv1, entry c of bn1, and input slice c of conv2, so tensor shapes stay
consistent and the residual (block input/output) dimensions are untouched. The result is a genuinely smaller network.
"""
import argparse

import torch

from common import (accuracy, blocks, build_model, count_macs, count_params, fit, get_device, load_cifar, load_model, save_fp,
                    save_log, seed_all, split_val)


def prune_channels(model, ratio):
    """Keep the (1-ratio) fraction of channels with the largest L1 filter norm in every block; returns (new_model, widths)."""
    widths = [max(1, round(b.conv1.out_channels * (1 - ratio))) for b in blocks(model)]
    new = build_model(widths)
    stem = {k: v for k, v in model.state_dict().items() if not k.startswith(("layer1", "layer2", "layer3", "layer4"))}
    new.load_state_dict(stem, strict=False)  # stem conv/bn and classifier are unchanged
    for old_b, new_b, w in zip(blocks(model), blocks(new), widths):
        keep = old_b.conv1.weight.detach().abs().sum((1, 2, 3)).topk(w).indices.sort().values
        new_b.conv1.weight.data = old_b.conv1.weight.data[keep].clone()
        for attr in ("weight", "bias", "running_mean", "running_var"):
            getattr(new_b.bn1, attr).data = getattr(old_b.bn1, attr).data[keep].clone()
        new_b.conv2.weight.data = old_b.conv2.weight.data[:, keep].clone()
        new_b.bn2.load_state_dict(old_b.bn2.state_dict())
        if old_b.downsample is not None:
            new_b.downsample.load_state_dict(old_b.downsample.state_dict())
    return new, widths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.5, help="fraction of inner channels removed per block")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    seed_all(0)
    dev = get_device()
    xtr, ytr, xte, yte = load_cifar()
    xtr, ytr, xval, yval = (t.to(dev) for t in split_val(xtr, ytr))
    xte, yte = xte.to(dev), yte.to(dev)

    base = load_model("fp32", "cpu")
    model, widths = prune_channels(base, args.ratio)
    p0, p1 = count_params(base)[0], count_params(model)[0]
    m0, m1 = count_macs(base), count_macs(model)
    acc0 = accuracy(model.to(dev), xval, yval)
    print(f"inner widths per block: {widths}")
    print(f"params {p0 / 1e6:.2f}M -> {p1 / 1e6:.2f}M ({100 * (1 - p1 / p0):.0f}% fewer), "
          f"MACs {m0 / 1e6:.0f}M -> {m1 / 1e6:.0f}M ({100 * (1 - m1 / m0):.0f}% fewer)")
    print(f"validation accuracy right after pruning (no fine-tune): {acc0:.2f}%")

    history = [{"tag": "structured", "epoch": 0, "train_loss": None, "val_acc": round(acc0, 2)}]
    fit(model, xtr, ytr, args.epochs, args.lr, eval_data=(xval, yval), smoothing=0.1, tag="structured", history=history)
    save_log(history, "structured")
    print(f"\nStructured-pruned test accuracy: {accuracy(model, xte, yte):.2f}%  -> checkpoints/pruned_structured.pth")
    save_fp(model.cpu(), "structured", widths)
