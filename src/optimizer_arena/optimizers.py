import numpy as np

from optimizer_arena.models import gradient, hessian, mse_loss


def gradient_descent(X, y, lr=0.01, n_iters=1000, theta0=None, return_path=False):
    """Run batch gradient descent. Return final theta and the loss history."""
    if theta0 is None:
        theta = np.zeros(X.shape[1])
    else:
        theta = np.array(theta0, dtype=float)

    history = [mse_loss(X, y, theta)]
    path = [theta.copy()]
    for _ in range(n_iters):
        theta = theta - lr * gradient(X, y, theta)
        history.append(mse_loss(X, y, theta))
        path.append(theta.copy())

    if return_path:
        return theta, history, np.array(path)
    return theta, history


def newton(X, y, n_iters=10, theta0=None, tol=1e-10, return_path=False):
    """Run Newton's method. Return final theta and the loss history."""
    if theta0 is None:
        theta = np.zeros(X.shape[1])
    else:
        theta = np.array(theta0, dtype=float)

    H = hessian(X)
    history = [mse_loss(X, y, theta)]
    path = [theta.copy()]
    for _ in range(n_iters):
        g = gradient(X, y, theta)
        if np.linalg.norm(g) < tol:
            break
        theta = theta - np.linalg.solve(H, g)
        history.append(mse_loss(X, y, theta))
        path.append(theta.copy())

    if return_path:
        return theta, history, np.array(path)
    return theta, history