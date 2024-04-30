import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess image
img = cv2.imread('./picture.png')  # return array(400 x 400 x 3)
img = cv2.resize(img, (200, 200))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load pixel data
data = pd.read_csv("./modified_data.csv")
df = pd.DataFrame(data)
pixel_columns = [f'pixel{i}' for i in range(784)]
pixel_data = df[pixel_columns].values.T / 10000

# Define activation function
def relu_func(m):
    return np.maximum(0, m)

# Convolution operation
def conv(kernel_size, img, s, p, active_func):
    kernel = np.random.randn(kernel_size, kernel_size)
    width, height = img.shape
    matrix = np.zeros((int((width - kernel_size + 2 * p) / s + 1), int((height - kernel_size + 2 * p) / s + 1)))
    for row in range(int((height - kernel_size + 2 * p) / s + 1)):
        for col in range(int((width - kernel_size + 2 * p) / s + 1)):
            matrix[row, col] = np.sum(img[row * s: row * s + kernel_size, col * s: col * s + kernel_size] * kernel)
            if active_func == "relu":
                matrix[row, col] = relu_func(matrix[row, col])
    return matrix

# Max pooling operation
def max_pooling(pooling_size, kernel_size, img, p, s, active_func):
    matrix = conv(kernel_size, img, s, p, active_func)
    result = np.zeros((int(matrix.shape[0] / pooling_size), int(matrix.shape[1] / pooling_size)))
    for row in range(int(matrix.shape[0] / pooling_size)):
        for col in range(int(matrix.shape[1] / pooling_size)):
            result[row, col] = np.max(matrix[row * pooling_size: (row + 1) * pooling_size,
                                     col * pooling_size: (col + 1) * pooling_size])
    return result

# Define model architecture
def model(img_gray):
    conv1 = max_pooling(kernel_size=3, img=img_gray, s=1, p=0, pooling_size=7, active_func="relu")
    conv2 = max_pooling(kernel_size=3, img=img_gray, s=1, p=0, pooling_size=7, active_func="relu")
    conv1_flat = conv1.flatten().reshape(784, 1)
    conv2_flat = conv2.flatten().reshape(784, 1)
    return conv1_flat, conv2_flat

# Define prediction class
class Predict:
    def __init__(self, img_gray, pixel_data, labels):
        conv1, conv2 = model(img_gray)
        self.x = pixel_data     
        self.input = np.array([conv1, conv2]).reshape(2, 784)  # return 2D [2x784]
        self.d = labels  # return [42000 x 1]
        self.m = len(self.d)  # return 42000

        self.weights_l1 = np.random.rand(self.x.shape[0], self.x.shape[0])  # return 784 x 784
        self.bias_l1 = np.random.rand(self.x.shape[0], 1)  # return 784 x 1

        self.weights_l2 = np.random.rand(self.x.shape[0], 1)  # return 784 x 1
        self.bias_l2 = np.random.rand(1, 1)  # return 1x1

    def sigmoid(self, net):
        return 1 / (1 + np.exp(-net))

    def sigmoid_derivative(self, net):
        return self.sigmoid(net) * (1 - self.sigmoid(net))

    def loss_func(self):
        loss = sum((self.d[i] - self.propagation(self.x[:, i])) ** 2 for i in range(self.m)) / self.m
        return loss

    def propagation(self, x):
        x = x.reshape(-1, 1)  # convert (784,0) to (784,1)

        # Hidden Layer
        self.net_h = np.dot(self.weights_l1.T, x) + self.bias_l1 # return [784 x 1]
        self.z = self.sigmoid(self.net_h)  # return [784 x 1]
        
        # Output Layer
        self.net_o = (np.dot(self.weights_l2.T, self.z) + self.bias_l2) / 100  # return 1 x 1
        self.y = self.net_o  # sigmoid [1 x 1]
        return self.y

    def backpropagation(self, d, x):
        x = x.reshape(-1, 1)  # return (784 x 1)

        learning_rate = 0.05
        gradient_o = np.multiply(d - self.y, 1)  # return [1x1]
        D_o = learning_rate * (gradient_o @ self.z.T)  # return [1x1] x [1x784] = [1x784]
        new_w2 = self.weights_l2 + D_o.T  # return [784x1]
        new_bias2 = self.bias_l2 + learning_rate * gradient_o  # return  [1x1]
        self.weights_l2 = new_w2  # update w2
        self.bias_l2 = new_bias2  # update b2

        gradient_h = np.multiply(self.weights_l2 @ gradient_o, self.sigmoid_derivative(self.net_h))  # return [784x1]  [784,1] = [784x1]
        D_h = learning_rate * gradient_h @ x.T  # return [784x784]
        new_w1 = self.weights_l1 + D_h.T  # return [784x784]
        new_bias1 = self.bias_l1 + learning_rate * gradient_h  # return  [1x1]
        self.weights_l1 = new_w1  # update w1
        self.bias_l1 = new_bias1  # update b1

    def errors(self, d):
        return 0.5 * (d - self.y) ** 2

    def training(self):
        epochs = 100
        epsilon = 0.0001
        loss = []
        for epoch in range(epochs):
            print("Epoch:", epoch)
            stop_training = False
            for k in range(self.m):
                self.propagation(self.x[:, k])
                e = self.errors(self.d[k]) - self.propagation(self.input[0])
                # loss.append(e)
                if e <= epsilon:
                    stop_training = True
                    break
                self.backpropagation(self.d[k], self.x[:, k])
            
            if stop_training:
                break

pre = Predict(img_gray, pixel_data, df.label.values)
pre.training()

cnv1, cnv2 = model(img_gray)
print(int(pre.propagation(cnv1)))
plt.imshow(cnv1.reshape((28,28)), cmap="gray")
plt.show()
