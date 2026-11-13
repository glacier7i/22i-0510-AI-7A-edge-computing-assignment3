"""Shared utilities: CIFAR-10 data, quantization-ready ResNet-18, training loop, checkpoint I/O."""
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.models.quantization import resnet18 as _quantizable_resnet18

for _cat in (DeprecationWarning, FutureWarning):  # torch.ao.quantization migration notices, numpy-2.4/torchvision noise
    warnings.filterwarnings("ignore", category=_cat)
warnings.filterwarnings("ignore", message=".*align=0.*")
warnings.filterwarnings("ignore", message=".*oneDNN backend.*")  # CPU has VNNI, so reduce_range=False is fine

ROOT = Path(__file__).resolve().parent
DATA_DIR, CKPT_DIR, RES_DIR = ROOT / "data", ROOT / "checkpoints", ROOT / "results"
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

# name -> checkpoint file. *_int8 files are TorchScript (self-contained); the rest are state_dict checkpoints.
CKPTS = {
    "fp32": "fp32.pth",
    "ptq_int8": "ptq_int8.pt",
    "qat_int8": "qat_int8.pt",
    "unstructured": "pruned_unstructured.pth",
    "structured": "pruned_structured.pth",
}


def get_device():
    """GPU when available, otherwise CPU (training then works but is much slower)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_backend():
    """Best available quantized-kernel backend on this machine (onednn on the development machine)."""
    engines = torch.backends.quantized.supported_engines
    return next(e for e in ("onednn", "x86", "fbgemm", "qnnpack") if e in engines)


def seed_all(seed=0):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- data
def normalize(x01):
    """float tensor in [0,1], NCHW -> normalized."""
    return (x01 - MEAN) / STD


def load_cifar():
    """Returns xtr, ytr, xte, yte as normalized float tensors / long labels (CPU).
    Uses data/cifar-10-batches-py if it exists; only if that folder is missing is CIFAR-10 downloaded (about 170 MB)."""
    missing = not (DATA_DIR / "cifar-10-batches-py").exists()
    if missing:
        print(f"CIFAR-10 not found in {DATA_DIR}; downloading it once with torchvision (about 170 MB)...", flush=True)
    out = []
    for train in (True, False):
        d = datasets.CIFAR10(DATA_DIR, train=train, download=missing)
        x = normalize(torch.from_numpy(d.data).permute(0, 3, 1, 2).float() / 255)
        out += [x.contiguous(), torch.tensor(d.targets)]
    return out


def split_val(x, y, n=5000, seed=0):
    """Hold out n training images as a validation set (used for per-epoch logging; the test set is kept for final numbers)."""
    p = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    return x[p[n:]], y[p[n:]], x[p[:n]], y[p[:n]]


def augment(x):
    """GPU-side random crop (pad 4, reflect) + horizontal flip, per-sample."""
    n, dev = x.size(0), x.device
    xp = F.pad(x, (4,) * 4, mode="reflect")
    ar = torch.arange(32, device=dev)
    i, j = torch.randint(0, 9, (n,), device=dev), torch.randint(0, 9, (n,), device=dev)
    out = xp[torch.arange(n, device=dev)[:, None, None], :, (i[:, None] + ar)[:, :, None], (j[:, None] + ar)[:, None, :]]
    out = out.permute(0, 3, 1, 2)  # advanced indexing moved the channel dim last
    flip = torch.rand(n, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


# ----------------------------------------------------------------------------- model
def blocks(model):
    return [b for layer in (model.layer1, model.layer2, model.layer3, model.layer4) for b in layer]


def build_model(widths=None):
    """CIFAR ResNet-18 (3x3 stem, no max-pool), built on torchvision's quantizable variant (random init, no download).
    widths: optional list of 8 inner-channel counts (conv1->conv2 of every BasicBlock) for structurally pruned nets."""
    m = _quantizable_resnet18(weights=None, quantize=False, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    for b, w in zip(blocks(m), widths or []):
        b.conv1 = nn.Conv2d(b.conv1.in_channels, w, 3, b.conv1.stride, 1, bias=False)
        b.bn1 = nn.BatchNorm2d(w)
        b.conv2 = nn.Conv2d(w, b.conv2.out_channels, 3, 1, 1, bias=False)
    return m


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return total, nonzero


@torch.no_grad()
def count_macs(model):
    """Dense multiply-accumulates of conv + linear layers for one 32x32 image (float models only)."""
    macs = [0]

    def hook(m, _, out):
        if isinstance(m, nn.Conv2d):
            macs[0] += out.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
        else:
            macs[0] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()(torch.zeros(1, 3, 32, 32, device=next(model.parameters()).device))
    [h.remove() for h in hs]
    return macs[0]


# ----------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model, x, device="cpu", bs=500):
    model.eval()
    return torch.cat([model(x[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(x), bs)])


def accuracy(model, x, y, device=None):
    device = device or next(model.parameters(), torch.empty(0)).device  # quantized models have no parameters -> CPU
    return (predict(model, x, device) == y.cpu()).float().mean().item() * 100


def fit(model, xtr, ytr, epochs, max_lr, eval_data=None, wd=5e-4, bs=256, amp=True, smoothing=0.0,
        on_epoch=None, tag="train", history=None):
    """SGD + one-cycle LR on GPU-resident data. eval_data (a validation set) is only logged, never used for selection.
    Per-epoch stats are appended to `history` (a list of dicts) when given."""
    dev = xtr.device
    model.to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, nesterov=True, weight_decay=wd)
    steps = len(xtr) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=epochs * steps, pct_start=0.25)
    crit = nn.CrossEntropyLoss(label_smoothing=smoothing)
    for ep in range(epochs):
        if on_epoch:
            on_epoch(ep)
        model.train()
        perm, loss_sum, t0 = torch.randperm(len(xtr), device=dev), 0.0, time.time()
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            with torch.autocast(dev.type, torch.bfloat16, enabled=amp):
                loss = crit(model(augment(xtr[idx])), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.detach()
        row = {"tag": tag, "epoch": ep + 1, "train_loss": round(loss_sum.item() / steps, 4)}
        msg = f"[{tag}] epoch {ep + 1:>2}/{epochs}  loss {row['train_loss']:.3f}"
        if eval_data:
            row["val_acc"] = round(accuracy(model, *eval_data, dev), 2)
            msg += f"  val_acc {row['val_acc']:.2f}%"
        print(f"{msg}  ({time.time() - t0:.0f}s)", flush=True)
        if history is not None:
            history.append(row)
    return model


def timeit(fn, n, warmup=20):
    """Wall-clock seconds of n calls to fn after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return ts


def measure(fn_single, fn_batch, bs=128, rounds=3):
    """(latency ms for one image, throughput img/s for a batch). Best of `rounds` rounds, each a median, to
    suppress CPU frequency / scheduler noise."""
    lat = min(float(np.median(timeit(fn_single, 200, 30))) for _ in range(rounds)) * 1e3
    thr = max(bs / float(np.median(timeit(fn_batch, 20, 5))) for _ in range(rounds))
    return lat, thr


# ----------------------------------------------------------------------------- checkpoints
def save_log(rows, name):
    RES_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RES_DIR / f"log_{name}.csv", index=False)


def save_fp(model, name, widths=None):
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "widths": widths}, CKPT_DIR / CKPTS[name])


def save_int8(model, name):
    """INT8 models are saved as TorchScript so they load and run without any class definitions."""
    CKPT_DIR.mkdir(exist_ok=True)
    torch.jit.save(torch.jit.trace(model.cpu().eval(), torch.randn(1, 3, 32, 32)), CKPT_DIR / CKPTS[name])


def load_model(name, device="cpu"):
    path = CKPT_DIR / CKPTS[name]
    if name.endswith("_int8"):  # quantized kernels exist for CPU only
        torch.backends.quantized.engine = pick_backend()
        return torch.jit.load(path, map_location="cpu").eval()
    ck = torch.load(path, map_location=device, weights_only=True)
    m = build_model(ck["widths"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
