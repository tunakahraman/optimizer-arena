import numpy as np


def predict(X, theta):
    """Return the model predictions X @ theta."""
    return X @ theta


def mse_loss(X, y, theta):
    """Return the mean squared error."""
    residual = predict(X, theta) - y
    return np.mean(residual ** 2)


def gradient(X, y, theta):
    """Return the gradient of the MSE loss with respect to theta."""
    n = X.shape[0]
    residual = predict(X, theta) - y
    return (2 / n) * (X.T @ residual)

def hessian(X):
    """Return the Hessian of the MSE loss (constant for linear regression)."""
    n = X.shape[0]
    return (2 / n) * (X.T @ X)