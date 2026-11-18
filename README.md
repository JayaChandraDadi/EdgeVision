# EdgeVision: quantized image classifier for edge devices

An ImageNet-pretrained **MobileNetV3-Small** fine-tuned on **CIFAR-10** in **PyTorch** with **data augmentation**, compressed with post-training **INT8 quantization**, exported to **ONNX**, and benchmarked for size, **inference speed** and accuracy on CPU with **ONNX Runtime**.

## Results

CIFAR-10 test set (all 10,000 images); 2-vCPU Intel Xeon @ 2.10 GHz.

| model | size | top-1 | batch-1 latency (1 thread) | batch-64 throughput (1 thread) |
|---|---:|---:|---:|---:|
| PyTorch FP32 (eager) | 6.24 MB | 91.45% | 3.90 ms | 684 img/s |
| ONNX Runtime FP32 | 6.12 MB | 91.45% | 1.06 ms | 1,287 img/s |
| **ONNX Runtime INT8** | **1.71 MB** | **90.91%** | **0.94 ms** | **1,405 img/s** |

- **3.6× smaller** model for a **0.54-point** accuracy drop. The INT8 model agrees with the FP32 model on 97.3% of test images.
- Exporting to ONNX Runtime alone is a **3.7×** batch-1 speed-up over PyTorch eager. INT8 adds roughly another 10–14% on a single thread.
- At 32×32 input this network needs only about 17.5M multiply-adds per image, so latency is bound by per-operator overhead rather than arithmetic. That limits INT8's speed advantage on this server CPU. The size reduction is the main win here: it matters most for flash- and memory-constrained edge targets. Full table, including the 2-thread runs, in [`results/benchmark.md`](results/benchmark.md).

**Quantization choices** ([`results/quantization_ablation.json`](results/quantization_ablation.json)):

| format | calibration | top-1 | size |
|---|---|---:|---:|
| QDQ | min-max | 90.69% | 1.85 MB |
| QDQ | percentile 99.999 | 90.87% | 1.85 MB |
| **QOperator** | **percentile 99.999** | **90.91%** | **1.71 MB** |

Percentile calibration clips rare activation outliers, so the remaining range gets finer resolution. It beats min-max by 0.2 points.

The same FP32 ONNX file also runs on [TinyInfer](https://github.com/JayaChandraDadi/TinyInfer), my C++ inference engine, with predictions identical to ONNX Runtime on all 10,000 test images.

## Pipeline

```
CIFAR-10 ─► augment ─► fine-tune MobileNetV3-Small (PyTorch) ─► FP32 ONNX ─► calibrate + INT8 PTQ ─► benchmark
            crop/flip/   ImageNet weights, 10 epochs, 91.45%     BN folded     1,024 train images      size · speed · acc
            Cutout
```

**Model** ([`edgevision/model.py`](edgevision/model.py)). MobileNetV3-Small with ImageNet-pretrained weights from [d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch) (MIT). Only the 1000-way head is re-initialised (10 classes). It is adapted for 32×32 input:
- the stem and first block run at stride 1, so the final feature map is 4×4 instead of collapsing to 1×1;
- squeeze-excite FC layers are rewritten as 1×1 convs, so the exported graph is static (Conv, HardSwish, HardSigmoid, Mul, Add, pooling, Gemm) with no `Shape`/`Reshape` chains;
- 1.53M parameters.

**Fine-tuning** ([`scripts/train.py`](scripts/train.py)). SGD with Nesterov momentum and a one-cycle LR schedule (peak 0.08), weight decay 5e-4, label smoothing 0.1, batch 128, 10 epochs, about 3 minutes per epoch on 2 CPU cores. Augmentation runs batched on tensors ([`edgevision/data.py`](edgevision/data.py)): reflect-pad-4 random crop, horizontal flip and 8×8 Cutout. Checkpoints are saved every epoch, so an interrupted run resumes exactly. Per-epoch accuracy is in [`results/train_log.csv`](results/train_log.csv): 83.2% after the first epoch, 91.45% after the tenth.

**Export** ([`scripts/export_onnx.py`](scripts/export_onnx.py)). Opset 17 with a dynamic batch axis; BatchNorm is folded into the preceding convs. The script checks ONNX Runtime against PyTorch (max abs difference 2e-6).

**Quantization** ([`scripts/quantize.py`](scripts/quantize.py)). ONNX Runtime static post-training quantization: per-channel symmetric int8 weights and per-tensor uint8 activations. Ranges are calibrated on 1,024 *training* images, so the test set never influences the quantized model.

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_pretrained.py               # ImageNet MobileNetV3-Small checkpoint -> weights/
python scripts/fetch_cifar10.py                  # CIFAR-10 -> data/cifar10/{train,test}/<class>/

python scripts/train.py --data data/cifar10 --epochs 10
python scripts/export_onnx.py                    # -> models/edgevision_fp32.onnx
python scripts/quantize.py --data data/cifar10   # -> models/edgevision_int8.onnx
python scripts/benchmark.py --data data/cifar10  # -> results/benchmark.{md,json}

python scripts/predict.py --model models/edgevision_int8.onnx some_image.jpg
```

The trained FP32 checkpoint and both ONNX models are committed in `models/`, so `predict.py` and `benchmark.py` work without retraining.

> Data note: the reported run used a public JPEG export of CIFAR-10 (same 50k/10k split). `fetch_cifar10.py` downloads the official release via torchvision, so numbers from a fresh run may differ by a few tenths of a point.

## Tests

```bash
pytest -q
```

The suite checks:
- output shapes and the 4×4 final resolution;
- that the pretrained backbone loads every tensor except the head;
- that augmentation is seeded and that its crops are true shifts;
- that the ONNX graph is static with BN folded, and that ONNX Runtime matches PyTorch;
- that quantization produces a runnable model more than 2.5× smaller.

## Layout

```
edgevision/  model.py · data.py
scripts/     fetch_pretrained.py · fetch_cifar10.py · train.py · export_onnx.py · quantize.py · benchmark.py · predict.py
models/      edgevision_fp32.pt · edgevision_fp32.onnx · edgevision_int8.onnx
results/     train_log.csv · benchmark.md/json · quantization_ablation.json
third_party/ licence for the pretrained MobileNetV3 weights
```
