import numpy as np

from optimizer_arena.models import gradient, mse_loss


def gradient_descent(X, y, lr=0.01, n_iters=1000, theta0=None):
    """Run batch gradient descent. Return final theta and the loss history."""
    if theta0 is None:
        theta = np.zeros(X.shape[1])
    else:
        theta = np.array(theta0, dtype=float)

    history = [mse_loss(X, y, theta)]
    for _ in range(n_iters):
        theta = theta - lr * gradient(X, y, theta)
        history.append(mse_loss(X, y, theta))

    return theta, history