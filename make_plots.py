"""Generate all report figures from the CSV logs in results/ into results/graphs/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator

from common import CKPT_DIR, RES_DIR

OUT = RES_DIR / "graphs"
LABELS = {"fp32": "FP32", "ptq_int8": "PTQ\nINT8", "qat_int8": "QAT\nINT8", "unstructured": "Unstruct.\npruned",
          "structured": "Struct.\npruned"}
COLORS = {"fp32": "#4C78A8", "ptq_int8": "#F58518", "qat_int8": "#54A24B", "unstructured": "#E45756", "structured": "#B279A2"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def curves(log, name, title, accent):
    """Training loss + validation accuracy per epoch."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(log.epoch, log.train_loss, color=accent, lw=2)
    ax[0].set(xlabel="Epoch", ylabel="Training loss", title="Training loss")
    ax[1].plot(log.epoch, log.val_acc, color=accent, lw=2, marker="o", ms=3)
    ax[1].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Validation accuracy")
    for a in ax:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title, y=1.03)
    return fig, ax


def bars(ax, models, values, title, ylabel, fmt="{:.2f}", ylim=None):
    b = ax.bar([LABELS[m] for m in models], values, color=[COLORS[m] for m in models], width=0.65)
    ax.bar_label(b, labels=[fmt.format(v) for v in values], fontsize=8, padding=2)
    ax.set(title=title, ylabel=ylabel)
    ax.tick_params(axis="x", labelsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


if __name__ == "__main__":
    # 1. FP32 baseline training
    log = pd.read_csv(RES_DIR / "log_fp32.csv")
    fig, _ = curves(log, "fp32", "FP32 baseline training (ResNet-18, CIFAR-10)", COLORS["fp32"])
    save(fig, "fig_baseline_training")

    # 2. QAT fine-tuning (val accuracy vs the FP32 baseline), observer / BN freeze marks
    q, fp_final = pd.read_csv(RES_DIR / "log_qat.csv"), pd.read_csv(RES_DIR / "log_fp32.csv").val_acc.iloc[-1]
    fig, ax = curves(q, "qat", "QAT fine-tuning (fake-quantized ResNet-18)", COLORS["qat_int8"])
    n = len(q)
    ax[1].axhline(fp_final, color=COLORS["fp32"], ls="--", lw=1.5, label=f"FP32 baseline ({fp_final:.2f}%)")
    ax[1].axvline(n * 5 // 8 + 0.5, color="gray", ls=":", lw=1.2)
    ax[1].axvline(n * 3 // 4 + 0.5, color="gray", ls="-.", lw=1.2)
    lo, top = min(q.val_acc.min(), fp_final) - 0.15, max(q.val_acc.max(), fp_final) + 0.12
    ax[1].set_ylim(lo, top)  # leave room below the lowest point and for the legend
    ax[1].annotate("observers frozen", (n * 5 // 8 + 0.5, lo), rotation=90, va="bottom", ha="right", fontsize=8)
    ax[1].annotate("BN stats frozen", (n * 3 // 4 + 0.5, lo), rotation=90, va="bottom", ha="right", fontsize=8)
    ax[1].legend(loc="lower left", fontsize=8)
    save(fig, "fig_qat_training")

    # 3. Iterative unstructured pruning trace
    u = pd.read_csv(RES_DIR / "log_unstructured.csv")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(u.sparsity, u.val_acc_after_prune, color="#999999", marker="s", lw=1.8, label="right after pruning")
    ax.plot(u.sparsity, u.val_acc_after_finetune, color=COLORS["unstructured"], marker="o", lw=2, label="after fine-tuning")
    ax.axhline(u.val_acc_after_finetune.iloc[0], color=COLORS["fp32"], ls="--", lw=1.3, label="FP32 baseline")
    ax.set(xlabel="Global sparsity (%)", ylabel="Validation accuracy (%)", title="Iterative unstructured pruning")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig_unstructured_pruning")

    # 4. Structured pruning: per-block widths + fine-tuning recovery
    widths = torch.load(CKPT_DIR / "pruned_structured.pth", map_location="cpu", weights_only=True)["widths"]
    orig = [64, 64, 128, 128, 256, 256, 512, 512]
    s = pd.read_csv(RES_DIR / "log_structured.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    x = range(len(widths))
    ax[0].bar([i - 0.2 for i in x], orig, 0.4, color=COLORS["fp32"], label="original")
    ax[0].bar([i + 0.2 for i in x], widths, 0.4, color=COLORS["structured"], label="pruned")
    ax[0].set(xticks=list(x), xticklabels=[f"B{i + 1}" for i in x], xlabel="BasicBlock", ylabel="Inner channels (conv1 out / conv2 in)",
              title="Channels per block")
    ax[0].legend(fontsize=8)
    ax[1].plot(s.epoch, s.val_acc, color=COLORS["structured"], lw=2, marker="o", ms=3)
    ax[1].axhline(fp_final, color=COLORS["fp32"], ls="--", lw=1.5, label=f"FP32 baseline ({fp_final:.2f}%)")
    ax[1].set(xlabel="Fine-tuning epoch (0 = right after pruning)", ylabel="Validation accuracy (%)", title="Accuracy recovery")
    ax[1].legend(fontsize=8, loc="lower right")
    save(fig, "fig_structured_pruning")

    # 5. Benchmark figures
    b = pd.read_csv(RES_DIR / "benchmark.csv")
    m = list(b.model)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    bars(ax[0], m, b.accuracy_pct, "Test accuracy (y-axis truncated)", "Accuracy (%)", ylim=(b.accuracy_pct.min() - 1, b.accuracy_pct.max() + 0.6))
    bars(ax[1], m, b.file_size_mb, "Checkpoint size", "Size (MB)", "{:.1f}", ylim=(0, b.file_size_mb.max() * 1.35))
    ax[1].scatter(range(len(m)), b.gzip_size_mb, color="black", marker="_", s=400, zorder=3, label="gzip-compressed size")
    ax[1].legend(fontsize=8, loc="upper right")
    save(fig, "fig_benchmark_accuracy_size")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    bars(ax[0], m, b.cpu_latency_ms_bs1, "CPU latency, batch 1 (4 threads)", "ms / image", "{:.1f}", ylim=(0, b.cpu_latency_ms_bs1.max() * 1.15))
    bars(ax[1], m, b.cpu_throughput_img_s_bs128, "CPU throughput, batch 128 (4 threads)", "images / s", "{:.0f}", ylim=(0, b.cpu_throughput_img_s_bs128.max() * 1.15))
    save(fig, "fig_benchmark_speed")

    # 6. ONNX Runtime
    o = pd.read_csv(RES_DIR / "onnx_benchmark.csv").set_index("onnx_model")
    o = o.loc[["fp32.onnx", "fp32_int8.onnx", "unstructured.onnx", "structured.onnx"]].reset_index()
    names = ["FP32", "FP32\nINT8 (ORT)", "Unstruct.\npruned", "Struct.\npruned"]
    cols = ["#4C78A8", "#72B7B2", "#E45756", "#B279A2"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"wspace": 0.4})
    for a, col, title, ylabel, fmt in [(ax[0], "accuracy_pct", "Test accuracy (y-axis truncated)", "Accuracy (%)", "{:.2f}"),
                                       (ax[1], "cpu_latency_ms_bs1", "Latency, batch 1", "ms / image", "{:.2f}"),
                                       (ax[2], "cpu_throughput_img_s_bs128", "Throughput, batch 128", "images / s", "{:.0f}")]:
        bb = a.bar(names, o[col], color=cols, width=0.65)
        a.bar_label(bb, labels=[fmt.format(v) for v in o[col]], fontsize=8, padding=2)
        a.set(title=title, ylabel=ylabel)
        a.tick_params(axis="x", labelsize=8)
        a.grid(axis="x", visible=False)
        a.set_ylim(*((o[col].min() - 1, o[col].max() + 0.5) if col == "accuracy_pct" else (0, o[col].max() * 1.15)))
    fig.suptitle("ONNX Runtime, CPU (4 threads)", y=1.03)
    save(fig, "fig_onnx")
    print("figures written to results/graphs")
