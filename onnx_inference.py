"""ONNX Runtime CPU inference: accuracy on the CIFAR-10 test set + latency/throughput for every exported ONNX model."""
import argparse
import os
import numpy as np
import onnxruntime as ort
import pandas as pd

from common import RES_DIR, load_cifar, measure
from export_onnx import ONNX_DIR

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    _, _, xte, yte = load_cifar()
    xte = xte.numpy()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = min(args.threads, os.cpu_count() or 1)
    rows = []
    for path in sorted(ONNX_DIR.glob("*.onnx")):
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        run = lambda x: sess.run(None, {"input": x})[0]
        pred = np.concatenate([run(xte[i:i + 500]).argmax(1) for i in range(0, len(xte), 500)])
        x1, xb = xte[:1], xte[:128]
        lat, thr = measure(lambda: run(x1), lambda: run(xb), 128)
        rows.append({"onnx_model": path.name, "accuracy_pct": round(float((pred == yte.numpy()).mean() * 100), 2),
                     "file_size_mb": round(path.stat().st_size / 2**20, 2),
                     "cpu_latency_ms_bs1": round(lat, 2), "cpu_throughput_img_s_bs128": round(thr)})
        print(f"{path.name:<18} acc {rows[-1]['accuracy_pct']:.2f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "onnx_benchmark.csv", index=False)
    print(f"\n(ONNX Runtime CPU, {args.threads} threads)")
    print(df.to_string(index=False))
