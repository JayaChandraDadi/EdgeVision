| engine | model size | top-1 acc | threads | batch-1 median ms | batch-1 p90 ms | batch-64 img/s |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 (eager) | 6.24 MB | 91.45% | 1 | 3.897 | 6.063 | 684 |
| PyTorch FP32 (eager) | 6.24 MB | 91.45% | 2 | 3.774 | 4.156 | 1221 |
| ONNX Runtime FP32 | 6.12 MB | 91.45% | 1 | 1.064 | 1.212 | 1287 |
| ONNX Runtime FP32 | 6.12 MB | 91.45% | 2 | 1.129 | 1.299 | 2382 |
| ONNX Runtime INT8 | 1.71 MB | 90.91% | 1 | 0.936 | 1.353 | 1405 |
| ONNX Runtime INT8 | 1.71 MB | 90.91% | 2 | 1.006 | 1.492 | 1908 |
| TinyInfer FP32 (C++) | 6.12 MB | 91.45% | 1 | 4.485 | 4.640 | 247 |
| TinyInfer FP32 (C++) | 6.12 MB | 91.45% | 2 | 3.586 | 4.057 | 341 |

Prediction agreement with ONNX Runtime FP32: PyTorch FP32 (eager) 100.00%, ONNX Runtime FP32 100.00%, ONNX Runtime INT8 97.32%, TinyInfer FP32 (C++) 100.00%
