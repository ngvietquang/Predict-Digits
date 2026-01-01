import numpy as np
from sklearn.datasets import fetch_openml
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

X_test = X[60000:]
y_test = y[60000:]

# Load model
nn = NeuralNetwork()
nn.load_model("Predict-Digits/train.npz")

# Accuracy
acc = nn.accuracy(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")


for i in range(5):
    img = X_test[i]
    label = y_test[i]

    pred = nn.forward_propagation(img)
    pred_label = np.argmax(pred)

    plt.imshow(img.reshape(28,28), cmap="gray")
    plt.title(f"True: {label} | Pred: {pred_label}")
    plt.axis("off")
    plt.show()
