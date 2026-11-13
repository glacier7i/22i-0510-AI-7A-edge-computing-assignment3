"""Automated benchmark of all five models on the CIFAR-10 test set.

Writes results/predictions.csv (per-image predictions of every model) and results/benchmark.csv (accuracy,
size, latency, throughput). Accuracy runs on CPU for every model so INT8 and FP32 are compared like for like.
"""
import argparse
import gzip
import os
from statistics import median
from time import perf_counter

import pandas as pd
import torch

from common import CKPT_DIR, CKPTS, RES_DIR, count_macs, count_params, load_cifar, load_model, measure, predict, timeit


def warm_up_cpu(seconds=3):
    """Spin the CPU up to its steady clock before timing so the first model is not penalised."""
    a, t0 = torch.randn(1024, 1024), perf_counter()
    while perf_counter() - t0 < seconds:
        a @ a


@torch.inference_mode()
def cpu_perf(model, bs=128):
    x1, xb = torch.randn(1, 3, 32, 32), torch.randn(bs, 3, 32, 32)
    return measure(lambda: model(x1), lambda: model(xb), bs)


@torch.inference_mode()
def gpu_latency(name):
    model, x = load_model(name, "cuda"), torch.randn(1, 3, 32, 32, device="cuda")

    def run():
        model(x)
        torch.cuda.synchronize()
    return median(timeit(run, 300)) * 1e3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for latency/throughput (edge-like setting)")
    ap.add_argument("--eval-threads", type=int, default=8, help="CPU threads for the full test-set accuracy pass")
    args = ap.parse_args()
    args.threads, args.eval_threads = (min(t, os.cpu_count() or 1) for t in (args.threads, args.eval_threads))

    RES_DIR.mkdir(exist_ok=True)
    _, _, xte, yte = load_cifar()
    preds, rows, fp_stats = pd.DataFrame({"index": range(len(yte)), "label": yte.tolist()}), [], {}
    torch.set_num_threads(args.threads)
    warm_up_cpu()

    for name, fname in CKPTS.items():
        path, quant = CKPT_DIR / fname, name.endswith("_int8")
        model = load_model(name, "cpu")

        torch.set_num_threads(args.eval_threads)
        preds[name] = predict(model, xte).tolist()
        acc = (preds[name] == yte.numpy()).mean() * 100

        if not quant:  # parameter / MAC counts are read off the float model; INT8 nets share fp32's architecture
            fp_stats[name] = (*count_params(model), count_macs(model))
        params, nonzero, macs = fp_stats["fp32" if quant else name]

        torch.set_num_threads(args.threads)
        lat, thr = cpu_perf(model)
        rows.append({
            "model": name, "accuracy_pct": round(acc, 2),
            "file_size_mb": round(path.stat().st_size / 2**20, 2),
            "gzip_size_mb": round(len(gzip.compress(path.read_bytes(), 6)) / 2**20, 2),
            "params_m": round(params / 1e6, 2), "nonzero_params_m": round(nonzero / 1e6, 2), "macs_m": round(macs / 1e6),
            "cpu_latency_ms_bs1": round(lat, 2), "cpu_throughput_img_s_bs128": round(thr),
            "gpu_latency_ms_bs1": None if quant or not torch.cuda.is_available() else round(gpu_latency(name), 2),
        })
        print(f"{name:<13} acc {acc:.2f}%  cpu {lat:.2f} ms  {thr:.0f} img/s", flush=True)

    preds.to_csv(RES_DIR / "predictions.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "benchmark.csv", index=False)
    print(f"\n(CPU threads for latency/throughput: {args.threads})")
    print(df.to_string(index=False))
