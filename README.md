# Assignment 3: Quantization and Pruning of ResNet-18 on CIFAR-10

Hamid Ashraf, Roll No. 22i-0510, Section 7A (Edge Computing & Intelligent Systems)

I trained a ResNet-18 on CIFAR-10 and then shrank it four ways: post-training quantization (PTQ), quantization-aware
training (QAT), unstructured pruning and structured pruning. All five models are compared by one benchmark script and
exported to ONNX. Everything is plain PyTorch. The report is in `report/`.

## Results (CIFAR-10 test set, 10,000 images)

| Model | Accuracy | Checkpoint size |
|---|---|---|
| FP32 baseline | 93.69 % | 42.7 MB |
| PTQ INT8 | 93.70 % | 10.9 MB |
| QAT INT8 | 93.69 % | 10.9 MB |
| Unstructured pruning (90 % of weights zero) | 93.53 % | 42.7 MB (7.7 MB after gzip) |
| Structured pruning (half the channels in each block) | 93.91 % | 21.7 MB |

The differences between these accuracies are smaller than the noise of a 10,000-image test set, so I read it as "no
accuracy lost". Details and plots are in the report.

## What you need
- Python 3.11 with `torch` 2.10, `torchvision` 0.25, `onnx`, `onnxruntime`, `pandas`, `matplotlib`, `pillow`, `numpy`.
- CIFAR-10 in `data/cifar-10-batches-py/`. The data folder is not included in the zip to keep it small. If the folder
  is missing, the first script that needs the data downloads CIFAR-10 automatically (about 170 MB, needs internet). If
  you already have the dataset, copy `cifar-10-batches-py` into a `data/` folder next to the scripts and nothing is
  downloaded.
- A CUDA GPU for training, fine-tuning and QAT. Without one the scripts fall back to the CPU, which is a lot slower.
  Converting to INT8 and running INT8 models always happens on the CPU.

The INT8 checkpoints were made with PyTorch 2.10 on an x86 CPU. If you use a different PyTorch version and they fail to
load, regenerate them by running `ptq.py` and `qat.py`.

## How to run
Run these in order from this folder:

```bash
python train_baseline.py      # trains the FP32 model       -> checkpoints/fp32.pth
python ptq.py                 # post-training quantization  -> checkpoints/ptq_int8.pt
python qat.py                 # quantization-aware training -> checkpoints/qat_int8.pt
python prune_unstructured.py  # iterative magnitude pruning -> checkpoints/pruned_unstructured.pth
python prune_structured.py    # channel pruning             -> checkpoints/pruned_structured.pth
python benchmark.py           # compares all five models    -> results/benchmark.csv, results/predictions.csv
python export_onnx.py         # ONNX export                 -> checkpoints/onnx/
python onnx_inference.py      # ONNX Runtime on the CPU     -> results/onnx_benchmark.csv
python make_plots.py          # figures                     -> results/graphs/
```

`pipeline.ipynb` runs the same steps and shows the output under each cell. By default it skips a training step if its
checkpoint already exists, so running all cells takes a few minutes. Set `RETRAIN = True` in the first cell to retrain
everything, which takes about 30 minutes on my laptop GPU (RTX 4060). Random seeds are fixed.

To classify images with any of the models:

```bash
python inference.py --model qat_int8 --n 8                      # first 8 test images
python inference.py --model structured --images cat.png dog.jpg # your own images
```

The model names are `fp32`, `ptq_int8`, `qat_int8`, `unstructured` and `structured`.

## Files
| File | What it does |
|---|---|
| `common.py` | data loading, the ResNet-18, the training loop, saving and loading checkpoints |
| `train_baseline.py` | trains the FP32 baseline |
| `ptq.py` | fuses layers, calibrates on 2,000 training images, converts to INT8 |
| `qat.py` | starts from the FP32 weights, fine-tunes with fake quantization, converts to INT8 |
| `prune_unstructured.py` | global L1 pruning in 5 steps up to 90 %, fine-tuning after each, then makes the masks permanent |
| `prune_structured.py` | removes the weakest channels inside each block and builds a smaller network |
| `inference.py` | loads and runs any of the five models |
| `benchmark.py` | accuracy, size, speed and per-image predictions for all five |
| `export_onnx.py`, `onnx_inference.py` | ONNX export and CPU inference with ONNX Runtime |
| `make_plots.py` | draws the figures used in the report |
| `checkpoints/` | the trained models (`.pth`, TorchScript `.pt` for INT8, and `onnx/`) |
| `results/` | CSV results, training logs and `graphs/` |
| `report/` | the technical report (PDF) |

## Things worth knowing
- The INT8 models are saved as TorchScript, so they load with `torch.jit.load` without needing the model code.
- Structured pruning only removes channels inside each residual block (the output of the first conv and the input of the
  second). The block input and output sizes stay the same so the skip connections still line up. The stem and the
  channels that run along the skips are not pruned. The checkpoint stores the new widths so the smaller network can be
  rebuilt when loading.
- Unstructured pruning leaves zeros inside normal dense tensors, so the file is not smaller and normal CPU kernels are
  not faster. The saving shows up in the gzip size and in the number of non-zero weights.
- PyTorch's eager INT8 models cannot be exported to ONNX. The INT8 ONNX model is made by ONNX Runtime's own quantizer
  from the FP32 model, so it is a separate model from the PTQ and QAT ones.
- Speed numbers use 4 CPU threads, and the accuracy pass uses 8, both limited to the number of cores the machine has.
  PyTorch CPU timings on my laptop changed a lot from run to run, so the ONNX Runtime timings are the ones to trust.
