import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self):
        self.weight1 = np.random.randn(128, 784) * np.sqrt(2/784) #(128, 784)
        self.bias1 = np.zeros((128, 1))  # (128, 1)

        self.weight2 = np.random.randn(64, 128) * np.sqrt(2/784) # (64, 128)
        self.bias2 = np.zeros((64, 1))  # (64, 1)

        self.weight3 = np.random.randn(10, 64)  * np.sqrt(2/784) # (10, 64)
        self.bias3 = np.zeros((10, 1))  # (10, 1)

    def relu(self, Z):
        return np.maximum(0, Z)
    def relu_prime(self, Z):
        return (Z > 0).astype(int)
    def softmax(self, Z):
        Z = np.clip(Z, -500, 500)
        Z -= np.max(Z, axis=0, keepdims=True)
        e_z = np.exp(Z)
        return e_z / np.sum(e_z, axis=0, keepdims=True)

    def forward_propagation(self, x_train):
        x_train = x_train.reshape(-1, 1)
        # Hidden Layer 1:
        self.net_h1 = np.dot(self.weight1, x_train) + self.bias1  
        self.z1 = self.relu(self.net_h1)  # (128, 1)

        # Hidden Layer 2:
        self.net_h2 = np.dot(self.weight2, self.z1) + self.bias2  
        self.z2 = self.relu(self.net_h2)  # (64, 1)

        # Output Layer:
        self.net_o = np.dot(self.weight3, self.z2) + self.bias3  
        self.y = self.softmax(self.net_o)  # (10, 1)

        return self.y

    def backward_propagation(self, learning_rate, x_train, y_train):
        x_train = x_train.reshape(-1, 1)

        # Output Layer:
        gradient_o = (self.y - y_train) 
        dw3 = np.dot(gradient_o, self.z2.T)  # (10, 64)
        self.weight3 -= learning_rate * dw3
        self.bias3 -= learning_rate * gradient_o

        # Hidden Layer 2:
        gradient_h2 = np.dot(self.weight3.T, gradient_o) * self.relu_prime(self.net_h2)
        dw2 = np.dot(gradient_h2, self.z1.T) # (64, 128)
        self.weight2 -= learning_rate * dw2
        self.bias2 -= learning_rate * gradient_h2

        # Hidden Layer 1:
        gradient_h1 = np.dot(self.weight2.T, gradient_h2) * self.relu_prime(self.net_h1)
        dw1 = np.dot(gradient_h1, x_train.T) # (128, 784)
        self.weight1 -= learning_rate * dw1
        self.bias1 -= learning_rate * gradient_h1  


    def cross_entropy_loss(self, y_true, y_pred):
        eps = 1e-9
        return -np.sum(y_true * np.log(y_pred + eps))
    
    def save_model(self, file_path):
        np.savez(file_path, W1=self.weight1, b1=self.bias1, 
                 W2=self.weight2, b2=self.bias2,
                 W3 = self.weight3, b3 = self.bias3)
        
    def load_model(self, file_path):
        with np.load(file_path) as data:
            self.weight1 = data['W1']
            self.bias1 = data['b1']
            self.weight2 = data['W2']
            self.bias2 = data['b2']
            self.weight3 = data['W3']
            self.bias3 = data['b3']
    def one_hot(self,y, num_classes=10):
        vec = np.zeros((num_classes, 1))
        vec[y] = 1
        return vec
    def predict(self, x):
        y_pred = self.forward_propagation(x)
        return np.argmax(y_pred)
    def accuracy(self, X, y):
        correct = 0
        n = X.shape[0]

        for x_i, y_i in zip(X, y):
            y_pred = self.predict(x_i)
            if y_pred == y_i:
                correct += 1
        return correct / n









