import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, img_path, data_path):
        self.img = self.load_and_preprocess_image(img_path)
        self.pixel_data = self.load_pixel_data(data_path)

    def load_and_preprocess_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (115, 115))
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def load_pixel_data(self, data_path):
        data = pd.read_csv(data_path)
        df = pd.DataFrame(data)
        pixel_columns = [f'pixel{i}' for i in range(784)]
        return df[pixel_columns].values.T 

    @staticmethod
    def relu_func(m):
        return np.maximum(0, m)

    @staticmethod
    def conv(kernel_size, img, s, p, active_func):
        kernel = np.random.randn(kernel_size, kernel_size)
        width, height = img.shape
        matrix = np.zeros((int((width - kernel_size + 2 * p) / s + 1), int((height - kernel_size + 2 * p) / s + 1)))
        for row in range(int((height - kernel_size + 2 * p) / s + 1)):
            for col in range(int((width - kernel_size + 2 * p) / s + 1)):
                matrix[row, col] = np.sum(img[row * s: row * s + kernel_size, col * s: col * s + kernel_size] * kernel)
                if active_func == "relu":
                    matrix[row, col] = ImageProcessor.relu_func(matrix[row, col])
        return matrix

    @staticmethod
    def max_pooling(pooling_size, kernel_size, img, p, s, active_func):
        matrix = ImageProcessor.conv(kernel_size, img, s, p, active_func)
        result = np.zeros((int(matrix.shape[0] / pooling_size), int(matrix.shape[1] / pooling_size)))
        for row in range(int(matrix.shape[0] / pooling_size)):
            for col in range(int(matrix.shape[1] / pooling_size)):
                result[row, col] = np.max(matrix[row * pooling_size: (row + 1) * pooling_size,
                                         col * pooling_size: (col + 1) * pooling_size])
        return result

    @classmethod
    def model(cls, img_gray):
        conv1 = cls.max_pooling(kernel_size=3, img=img_gray, s=1, p=0, pooling_size=4, active_func="relu")
        conv2 = cls.max_pooling(kernel_size=3, img=img_gray, s=1, p=0, pooling_size=4, active_func="relu")
        conv1_flat = conv1.flatten().reshape(784, 1)
        conv2_flat = conv2.flatten().reshape(784, 1)
        return conv1_flat, conv2_flat

# Example usage

