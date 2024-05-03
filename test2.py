import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from convalution import ImageProcessor

class Predict:
    def __init__(self, img_path, data_path):
        self.processor = ImageProcessor(img_path=img_path, data_path=data_path)
        self.data = pd.read_csv(data_path)
        self.df = pd.DataFrame(self.data)
        self.conv1, self.conv2 = self.processor.model(self.processor.img)
        self.x = self.processor.pixel_data
        self.input = np.array([self.conv1, self.conv2]).reshape(2, 784)
        self.d = self.df.label
        self.m = len(self.d)

        self.weights_l1 = np.random.randn(self.x.shape[0], self.x.shape[0]) * 0.01
        self.bias_l1 = np.zeros((self.x.shape[0], 1))

        self.weights_l2 = np.random.randn(self.x.shape[0], 10) * 0.01
        self.bias_l2 = np.zeros((10, 1))

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)

    def loss_func(self, y_pred, y_true):
        m = y_pred.shape[1]
        return -np.sum(np.multiply(y_true, np.log(y_pred))) / m

    def propagation(self, x):
        x = x.reshape(-1, 1)

        self.net_h = np.dot(self.weights_l1.T, x) + self.bias_l1
        self.z = 1 / (1 + np.exp(-self.net_h))

        self.net_o = np.dot(self.weights_l2.T, self.z) + self.bias_l2
        self.y = self.softmax(self.net_o)
        return self.y

    def backpropagation(self, y_pred, y_true, x):
        x = x.reshape(-1, 1)
        m = y_pred.shape[1]

        dZ2 = y_pred - y_true
        dW2 = (1 / m) * np.dot(self.z, dZ2.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.weights_l2, dZ2) * (self.z * (1 - self.z))
        dW1 = (1 / m) * np.dot(x, dZ1.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.weights_l2 -= dW2
        self.bias_l2 -= db2
        self.weights_l1 -= dW1
        self.bias_l1 -= db1

    def predict(self, x):
        return np.argmax(self.propagation(x))

    def errors(self, y_pred, y_true):
        return 0.5 * np.sum((y_true - y_pred) ** 2)

    def training(self, learning_rate=0.01, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            for i in range(self.m):
                x = self.x[:, i]
                y_true = np.zeros((10, 1))
                y_true[self.d[i]] = 1

                y_pred = self.propagation(x)
                loss = self.loss_func(y_pred, y_true)
                total_loss += loss

                self.backpropagation(y_pred, y_true, x)

                if self.predict(y_pred) == self.d[i]:
                    correct_predictions += 1

            avg_loss = total_loss / self.m
            accuracy = correct_predictions / self.m * 100
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        self.save_parameters()

    def save_parameters(self, file_path="parameters.txt"):
        with open(file_path, 'w') as f:
            f.write("weights_l1:\n")
            np.savetxt(f, self.weights_l1, delimiter=',')
            f.write("\n\n")

            f.write("bias_l1:\n")
            np.savetxt(f, self.bias_l1, delimiter=',')
            f.write("\n\n")

            f.write("weights_l2:\n")
            np.savetxt(f, self.weights_l2, delimiter=',')
            f.write("\n\n")

            f.write("bias_l2:\n")
            np.savetxt(f, self.bias_l2, delimiter=',')
            f.write("\n\n")

    def show_img(self):
        plt.imshow(self.input[0].reshape((28, 28)), cmap="gray")
        plt.show()

img_path = './picture.png'
data_path = './modified_data.csv'

predictor = Predict(img_path, data_path)
predictor.show_img()
predictor.training()
