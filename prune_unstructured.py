"""Q2.2 / Q2.4 - Iterative global magnitude (L1) unstructured pruning with masking, fine-tuning, and permanent removal."""
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from common import accuracy, count_params, fit, get_device, load_cifar, load_model, save_fp, save_log, seed_all, split_val

SCHEDULE = [0.40, 0.60, 0.75, 0.85, 0.90]  # cumulative sparsity targets, one pruning iteration each

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-epochs", type=int, default=6, help="fine-tuning epochs after every pruning iteration")
    ap.add_argument("--lr", type=float, default=0.02)
    args = ap.parse_args()

    seed_all(0)
    dev = get_device()
    xtr, ytr, xte, yte = load_cifar()
    xtr, ytr, xval, yval = (t.to(dev) for t in split_val(xtr, ytr))
    xte, yte = xte.to(dev), yte.to(dev)

    model = load_model("fp32", dev)
    targets = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    n_target = sum(getattr(m, n).numel() for m, n in targets)

    def sparsity():
        return 100 * sum(int((getattr(m, n) == 0).sum()) for m, n in targets) / n_target

    base = accuracy(model, xval, yval)
    log, prev = [dict(iteration=0, target_sparsity=0.0, sparsity=sparsity(), val_acc_after_prune=base,
                      val_acc_after_finetune=base)], 0.0
    for k, target in enumerate(SCHEDULE, 1):
        # amount is a fraction of the *remaining* weights, so convert cumulative target -> incremental amount
        prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=(target - prev) / (1 - prev))
        acc_pruned = accuracy(model, xval, yval)
        fit(model, xtr, ytr, args.ft_epochs, args.lr, eval_data=(xval, yval), tag=f"prune {k}/{len(SCHEDULE)} @ {target:.0%}")
        log.append(dict(iteration=k, target_sparsity=target * 100, sparsity=sparsity(), val_acc_after_prune=acc_pruned,
                        val_acc_after_finetune=accuracy(model, xval, yval)))
        prev = target

    for m, n in targets:  # make pruning permanent: weight = weight_orig * mask, then drop mask/reparametrisation
        prune.remove(m, n)

    total, nonzero = count_params(model)
    log = pd.DataFrame(log).round(2)
    save_log(log, "unstructured")
    print(log.to_string(index=False))
    print(f"\nFinal: {sparsity():.1f}% sparse, {nonzero:,}/{total:,} non-zero params, "
          f"test acc {accuracy(model, xte, yte):.2f}%  -> checkpoints/pruned_unstructured.pth")
    save_fp(model.cpu(), "unstructured")
