"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np

def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test

def load_fashion_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, y_train, x_test, y_test):
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    num_classes = 10
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]

    return x_train, y_train, x_test, y_test

