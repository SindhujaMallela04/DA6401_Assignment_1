"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

def cross_entropy_loss(y_true, y_pred) :
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1. - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def cross_entropy_derivative(y_true, y_pred) :
    return (y_pred - y_true) / y_true.shape[0]
    # return y_pred - y_true

def MSE_Loss(y_true, y_pred) :
    return np.mean((y_true - y_pred) ** 2)

def MSE_Loss_derivative(y_true, y_pred) :
    return 2 * (y_pred - y_true) / y_true.shape[0]
