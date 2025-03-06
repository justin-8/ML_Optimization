Project: GPU-Accelerated MNIST Inference
Overview: Trained a 97.41%-accurate MLP on MNIST using Colab’s GPU, then unleashed batch inference at peak performance with PyTorch.
Results:
GPU throughput: 2,156,876 images/second (0.0046s avg for 10,000 images).
CPU throughput: 35,399 images/second (0.2825s).
Speedup: 5992% (60x faster).
Tools: PyTorch, Colab GPU.