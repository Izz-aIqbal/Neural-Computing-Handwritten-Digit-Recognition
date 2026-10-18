# Neural Computing System for Handwritten Digit Recognition

A collection of foundational Artificial Neural Network (ANN) architectures implemented completely from scratch using **pure NumPy**, designed for handwritten digit classification and topological analysis on the MNIST dataset.

---

## Executive Summary

This project explores the historical and mathematical evolution of neural computing by implementing four classic neural network models from first principles without relying on high-level deep learning frameworks (such as PyTorch or TensorFlow). The system processes the MNIST benchmark dataset (70,000 grayscale images of digits 0 through 9, sized 28x28 pixels split into 60,000 training and 10,000 testing samples) across four distinct computational paradigms.

---

## Implemented Neural Models

### 1. Perceptron (1958)
* **Architecture:** Single-layer binary classifier using a step activation function.
* **Key Focus:** Investigates linear separability limits and demonstrates why single-layer networks fail on non-linear problems like the XOR logic gate (achieving random-guess baseline performance).

### 2. Adaline - Adaptive Linear Neuron (1960)
* **Architecture:** Single-layer linear neuron utilizing continuous activation during training.
* **Learning Rule:** Implements gradient descent minimization of Mean Squared Error (MSE) via the Widrow-Hoff learning rule:
  $$\Delta w = \eta \cdot (target - prediction) \cdot input$$

### 3. Kohonen Self-Organizing Map - SOM (1982)
* **Architecture:** Unsupervised 2D grid topology ($12 \times 12 = 144$ neurons) mapped to 784 input pixel dimensions.
* **Capabilities:** Performs unsupervised feature clustering, prototype weight map generation, U-Matrix distance visualization, and anomaly detection using quantization errors at the 95th percentile threshold.
* **Performance Metrics:** Quantization Error (~0.25), Topographic Error (~4.12%), Precision (~85%), and Recall (~87%).

### 4. Multi-Layer Perceptron with Backpropagation (1986)
* **Architecture:** Deep feedforward network structure ($784 \rightarrow 128 \rightarrow 64 \rightarrow 10$) mapping pixel inputs to 10 digit classes.
* **Learning Mechanism:** Implements forward matrix transformations utilizing the **ReLU** activation function (avoiding vanishing gradients) followed by backward error propagation via the chain rule to update network weights and biases.

---

## Repository Directory Structure

```text
.
├── data/
│   └── mnist/
├── models/
│   ├── perceptron.py
│   ├── adaline.py
│   ├── som.py
│   └── mlp.py
├── utils/
│   ├── preprocessing.py
│   └── metrics.py
├── notebooks/
│   └── evaluation_experiments.ipynb
├── requirements.txt
└── README.md
