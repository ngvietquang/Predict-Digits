import numpy as np
import pandas as pd
from convalution import ImageProcessor
import matplotlib.pyplot as plt
# Load data
img_path = './picture.png'
data_path = './train.csv'
data = pd.read_csv(data_path)
df = pd.DataFrame(data)
df.fillna(0, inplace=True)
y_train = df['label'].values
x_train = df.drop(columns=['label']).values
processor = ImageProcessor(img_path=img_path, data_path=data_path)
class NeuralNetwork:
    def __init__(self):
        self.conv1, self.conv2 = processor.model(processor.img)
        self.y_train = y_train  # (42000,)
        self.m = len(self.y_train)
        self.x_train = x_train.T/255.0  # (784, 42000)
        self.y = None

        self.weight1 = np.random.randn(784, 784)*np.sqrt(2/42000) # (784, 784)
        self.bias1 = np.random.rand(784, 1)  # (784, 1)

        self.weight2 = np.random.randn(784, 784)*np.sqrt(2/42000)  # (784, 784)
        self.bias2 = np.random.rand(784, 1)  # (784, 1)

        self.weight3 = np.random.randn(784, 10)*np.sqrt(2/42000)  # (784, 10)
        self.bias3 = np.random.rand(10, 1)   # (10, 1)

    def tanh(self,Z):
        return np.tanh(Z)
    def tanh_prime(self,Z):
        return 1 - self.tanh(Z)**2
    def relu(self, Z):
        return np.maximum(0, Z)
    def relu_prime(self, Z):
        return (Z > 0).astype(int)
    def softmax(self, Z):
        Z -= np.max(Z, axis=0, keepdims=True)  # Shift Z to avoid large values
        e_z = np.exp(Z)
        return e_z / np.sum(e_z, axis=0, keepdims=True)

    def forward_propagation(self, x_train):
        x_train = x_train.reshape(-1, 1)
        # Hidden Layer 1:
        self.net_h1 = np.dot(self.weight1.T, x_train) + self.bias1  # (784, 1)
        self.z1 = self.tanh(self.net_h1)  # (784, 1)

        # Hidden Layer 2:
        self.net_h2 = np.dot(self.weight2.T, self.z1) + self.bias2  # (784, 1)
        self.z2 = self.relu(self.net_h2)  # (784, 1)

        # Output Layer:
        self.net_o = np.dot(self.weight3.T, self.z2) + self.bias3  # (10, 1)
        self.y = self.softmax(self.net_o)  # (10, 1)

        return self.y

    def backward_propagation(self, learning_rate, x_train, y_train):
        x_train= x_train.reshape(-1, 1)

        # Output Layer:
        gradient_o = ((self.y - y_train)/self.m).astype(np.float64)
        D_o = (learning_rate * np.dot(self.z2, gradient_o.T))
        self.weight3 -= D_o
        self.bias3 -= learning_rate * gradient_o

        # Hidden Layer 2:
        gradient_h2 = (np.dot(self.weight3, gradient_o) * self.relu_prime(self.net_h2))/self.m
        D_h2 = (learning_rate * np.dot(self.z1, gradient_h2.T)).astype(np.float64)
        self.weight2 -= D_h2
        self.bias2 -= learning_rate * gradient_h2

        # Hidden Layer 1:
        gradient_h1 = (np.dot(self.weight2, gradient_h2) * self.tanh_prime(self.net_h1))/self.m
        D_h1 = (learning_rate * np.dot(x_train, gradient_h1.T)).astype(np.float64)
        self.weight1 -= D_h1
        self.bias1 -= learning_rate * gradient_h1
    def loss(self, y_pred):
        return np.mean((y_pred - self.y_train) ** 2)
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
    def fit(self, learning_rate, epochs, batch_size, x_train, y_train):
        self.loss_lst = []
        num_samples = len(x_train)

        for epoch in range(epochs):
            epoch_loss = 0
            print(f'Epoch {epoch+1}/{epochs}')

            shuffled_indices = np.random.permutation(num_samples)
            x_train_shuffled = x_train[shuffled_indices]
            y_train_shuffled = y_train[shuffled_indices]

            for i in range(0, num_samples, batch_size):
                x_batch = x_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
            for i in range(x_batch.shape[0] - 1):
                y_pred = self.forward_propagation(x_batch[i])
                self.backward_propagation(learning_rate, x_batch[i], y_batch[i])
                batch_loss = self.loss(y_pred)
                epoch_loss += batch_loss

            epoch_loss /= (num_samples / batch_size)
            print(f"Train Loss: {epoch_loss}")

            self.loss_lst.append(epoch_loss)

        self.save_model("./train.npz")

    def predict(self, X):
        self.forward_propagation(X)
        return np.argmax(self.y, axis=0)
    def show_img(self):
        plt.imshow(self.conv1.reshape((28, 28)), cmap="gray")
        plt.show()


# nn = NeuralNetwork()
# pic = ImageProcessor(img_path=img_path, data_path= data_path)
# img_gray = pic.img

# cv1,cv2  = pic.model(img_gray)
# nn.load_model("./train.npz")
# nn.fit(0.1, 100, 32, x_train, y_train)


