import numpy as np
from sklearn.datasets import fetch_openml
from NeuralNetwork import NeuralNetwork

# Load MNIST
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data.astype(np.float32) / 255.0 #(70000, 784)
y = mnist.target.astype(int) #(70000,1)

X_train = X[:60000] 
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]


# Init model
nn = NeuralNetwork()

# Train
epochs = 100
batch_size = 64
lr = 0.0001
n = X_train.shape[0]

for epoch in range(epochs):
    perm = np.random.permutation(n)
    X_train = X_train[perm]
    y_train = y_train[perm]

    total_loss = 0

    for i in range(0, n, batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        for x_i, y_i in zip(X_batch, y_batch):
            y_true = nn.one_hot(y_i)
            y_pred = nn.forward_propagation(x_i)

            total_loss += nn.cross_entropy_loss(y_true, y_pred)
            nn.backward_propagation(lr, x_i, y_true)
    acc = nn.accuracy(X_test, y_test)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/n:.4f} - Accuracy: {acc*100:.2f}%")
    if acc >= 0.95:
        print("Stop training as accuracy reached 95%")
        nn.save_model("train.npz")
        break

nn.save_model("train.npz")
print("Model saved!")
