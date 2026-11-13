"""ONNX deployment - export the FP32 / unstructured / structured models to ONNX, plus an INT8 (QDQ) ONNX of the FP32
model made with ONNX Runtime's static quantizer. PyTorch eager-mode INT8 graphs do not export to ONNX, so the INT8
ONNX model is quantized on the ONNX side, using a calibration subset of the training data."""
import logging

import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from common import CKPT_DIR, load_cifar, load_model

ONNX_DIR = CKPT_DIR / "onnx"


class Calib(CalibrationDataReader):
    def __init__(self, x, bs=50, n=20):
        self.it = iter(x[:bs * n].split(bs))

    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {"input": b.numpy()}


if __name__ == "__main__":
    ONNX_DIR.mkdir(exist_ok=True)
    xtr, *_ = load_cifar()
    xtr = xtr[torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0))]

    for name in ("fp32", "unstructured", "structured"):
        torch.onnx.export(load_model(name, "cpu"), torch.randn(1, 3, 32, 32), ONNX_DIR / f"{name}.onnx",
                          input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
        print(f"exported {name}.onnx")

    logging.getLogger().setLevel(logging.ERROR)  # silence the optional "run pre-processing" advisory
    quantize_static(ONNX_DIR / "fp32.onnx", ONNX_DIR / "fp32_int8.onnx", Calib(xtr), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    print("exported fp32_int8.onnx (ONNX Runtime static INT8)")
