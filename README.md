# 🧠 Handwritten Digit Recognition from Scratch (NN + Web)

## 📌 Overview
This project implements a **handwritten digit recognition system from scratch**, without using deep learning frameworks such as TensorFlow or PyTorch.

The system includes:
- A fully **self-implemented Neural Network** (NumPy only)
- **Backpropagation + Mini-batch Gradient Descent**
- **Computer Vision preprocessing** to match MNIST format
- A **Flask backend API**
- An interactive **web drawing interface (HTML + Canvas)**

The trained model achieves **~95% accuracy on MNIST** and correctly predicts handwritten digits drawn by users in real time.

---

## ✨ Key Features
- Neural Network architecture: **784 → 128 → 64 → 10**
- Activation functions:
  - ReLU (hidden layers)
  - Softmax (output layer)
- Loss function: **Categorical Cross-Entropy**
- Training techniques:
  - Mini-batch training
  - Early stopping at 95% accuracy
- Image preprocessing:
  - Thresholding
  - Bounding box extraction
  - Resize + padding to **28×28**
  - Normalization
- Web demo with confidence score
- Model saving & loading (`.npz`)

---

## 🧠 Neural Network Architecture

**Why this architecture?**
- 784 matches MNIST input size
- Gradual reduction (128 → 64) helps feature abstraction
- Lightweight but powerful enough to reach high accuracy
- Suitable for training from scratch without GPU

---

## 🧮 Training Results
- Dataset: **MNIST**
- Training samples: 60,000
- Optimizer: Gradient Descent (manual backprop)
- Batch size: 64
- Learning rate: tuned manually
- Final result:
  - **Accuracy ≈ 95%**
  - Training stopped automatically when reaching threshold

Example log:

---

## 🚀 How to Run
```bash
python app.py

