# Federated Learning Simulation with MobileNetV2 on CIFAR-10

This project simulates a peer-to-peer (P2P) federated learning setup using MobileNetV2 on the CIFAR-10 dataset. We trained multiple node models, combined them into a global model, quantized it for efficiency, and visualized the results. Despite some quantization hiccups, we landed a solid solution with clear insights into accuracy, size, and training performance.

## Objective
The goal was to:
- Train 5 node models on CIFAR-10.
- Average them into a global model to mimic P2P federated learning.
- Quantize the global model to INT8 for reduced size and efficiency.
- Compare accuracy and size between the global and quantized models.
- Visualize training loss, accuracy, and size differences.

##Project Steps
##Step 1: Setup and Data Preparation
We started by loading MobileNetV2 with pretrained weights, tweaking its classifier for CIFAR-10’s 10 classes, and setting up data loaders for training and testing. Everything ran smoothly on Colab’s T4 GPU, with no issues to report.

##Step 2: Training Node Models
Five node models were trained on CIFAR-10 for one epoch each, using a standard optimizer and loss function. Training losses dropped from around 1.7 to 0.6–0.7 per node, giving us a solid dataset for later visualization. No major problems here—just steady progress.

##Step 3: Averaging Models (P2P Simulation)
To simulate P2P federation, we averaged the weights of the 5 node models into a single global_model. Initially, we hit a snag with a namespace mix-up (calling the wrong library), and later, averaging failed due to incompatible data types (some integers couldn’t be averaged). We fixed this by correcting the library reference and skipping non-float weights, resulting in a working global_model.

##Step 4: Testing the Global Model
We tested the global_model on the test set, leveraging the T4 GPU. It scored an impressive 88.57% accuracy right out of the gate—no hiccups, just a strong baseline.

##Step 5: Quantization


Initial Plan: ONNX Runtime
We exported the model to ONNX format and tried quantizing it to INT8 using ONNX Runtime’s tools.
Problem: The quantized model used an operation (ConvInteger) that ONNX Runtime didn’t support on CPU or even GPU (after installing GPU support).

Attempts to Fix:
- Switched to GPU execution—still failed.
- Updated the ONNX version and tweaked quantization settings—no luck.
- Added a preprocessing step (suggested by ONNX docs) to simplify the model—hit a dead end with API mismatches.

Outcome: ONNX Runtime couldn’t run the quantized model, so we pivoted.

Final Solution: PyTorch Quantization
- We switched to PyTorch’s built-in dynamic quantization, targeting convolutional and linear layers for INT8.
- Problem 1: GPU quantization failed because PyTorch’s quantized operations aren’t supported on CUDA.
- Problem 2: Running on CPU hit a tensor mismatch (CPU inputs vs. GPU weights).
- Fix: Moved the model to CPU before and after quantization, ensuring everything stayed aligned.
Outcome: Success! Quantized accuracy hit 88.64%, with sizes at 8.8 MB (global) and 8.7 MB (quantized).