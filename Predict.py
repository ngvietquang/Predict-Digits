import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from convalution import ImageProcessor

img_path = './picture.png'
data_path = './modified_data.csv'
processor = ImageProcessor(img_path=img_path, data_path=data_path)

data =   pd.read_csv(data_path)
df = pd.DataFrame(data)

class Predict:
    def __init__(self):
        conv1, conv2 = processor.model(processor.img)
        self.x = processor.pixel_data
        self.input = np.array([conv1, conv2]).reshape(2, 784)
        self.d = df.label
        self.m = len(self.d)

        self.weights_l1 = np.random.rand(self.x.shape[0], self.x.shape[0])
        self.bias_l1 = np.random.rand(self.x.shape[0], 1)

        self.weights_l2 = np.random.rand(self.x.shape[0], 1)
        self.bias_l2 = np.random.rand(1, 1)

    def sigmoid(self, net):
        return 1 / (1 + np.exp(-net))

    def sigmoid_derivative(self, net):
        return self.sigmoid(net) * (1 - self.sigmoid(net))

    def loss_func(self):
        loss = sum((self.d[i] - self.propagation(self.x[:, i])) ** 2 for i in range(self.m)) / self.m
        return loss

    def propagation(self, x):
        x = x.reshape(-1, 1)

        self.net_h = np.dot(self.weights_l1.T, x) + self.bias_l1
        self.z = self.sigmoid(self.net_h)

        self.net_o = (np.dot(self.weights_l2.T, self.z) + self.bias_l2) 
        self.y = self.net_o
        return self.y

    def backpropagation(self, d, x):
        x = x.reshape(-1, 1)

        learning_rate = 0.01
        gradient_o = np.multiply(d - self.y, 1)
        D_o = learning_rate * (gradient_o @ self.z.T)
        new_w2 = self.weights_l2 + D_o.T
        new_bias2 = self.bias_l2 + learning_rate * gradient_o
        self.weights_l2 = new_w2
        self.bias_l2 = new_bias2

        gradient_h = np.multiply(self.weights_l2 @ gradient_o, self.sigmoid_derivative(self.net_h))
        D_h = learning_rate * gradient_h @ x.T
        new_w1 = self.weights_l1 + D_h.T
        new_bias1 = self.bias_l1 + learning_rate * gradient_h
        self.weights_l1 = new_w1
        self.bias_l1 = new_bias1

    def predict(self):
        return self.propagation(self.input[0])

    def errors(self, d):
        return 0.5 * (d - self.y) ** 2

    def training(self):
        epochs = 100
        epsilon = 0.0000001

        with open("./train.txt", 'w') as f:
            for epoch in range(epochs):
                print("Epoch:", epoch)
                for k in range(self.m):
                    self.propagation(self.x[:, k])
                    e = self.errors(self.d[k])
                    # print(self.predict())
                    if e <= epsilon: 
                        print(self.predict())
                        f.write("Epoch %d\n" % epoch)
                        f.write("Result 1: %d\n" % self.propagation(self.input[0]).item())
                        f.write("Result 2: %d\n" % self.propagation(self.input[1]).item())



                    
                        f.write("Weights L1:\n")
                        np.savetxt(f, self.weights_l1)

                        f.write("Weights L2:\n")
                        np.savetxt(f, self.weights_l2)

                        f.write("Bias L1:\n")
                        np.savetxt(f, self.bias_l1)

                        f.write("Bias L2:\n")
                        np.savetxt(f, self.bias_l2)



                self.backpropagation(self.d[k], self.x[:, k])


    def show_img(self):
        plt.imshow(self.input[0].reshape((28, 28)), cmap="gray")
        plt.show()


predictor = Predict()
predictor.show_img()
predictor.training()