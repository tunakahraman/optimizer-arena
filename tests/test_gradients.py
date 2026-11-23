import numpy as np

from optimizer_arena.data import add_bias_column, make_linear_data
from optimizer_arena.models import gradient, hessian, mse_loss


def _setup():
    x, y = make_linear_data()
    return add_bias_column(x), y


def test_gradient_matches_finite_differences():
    X, y = _setup()
    theta = np.array([-0.5, 1.5])  # (b, w)
    eps = 1e-6
    numerical = np.zeros_like(theta)
    for i in range(len(theta)):
        step = np.zeros_like(theta)
        step[i] = eps
        numerical[i] = (mse_loss(X, y, theta + step) - mse_loss(X, y, theta - step)) / (2 * eps)
    assert np.allclose(gradient(X, y, theta), numerical, rtol=1e-5, atol=1e-6)


def test_hessian_matches_finite_differences_of_gradient():
    X, y = _setup()
    theta = np.array([-0.5, 1.5])
    eps = 1e-6
    H_num = np.zeros((2, 2))
    for i in range(2):
        step = np.zeros(2)
        step[i] = eps
        H_num[:, i] = (gradient(X, y, theta + step) - gradient(X, y, theta - step)) / (2 * eps)
    assert np.allclose(hessian(X), H_num, rtol=1e-5, atol=1e-6)


def test_newton_one_step_reaches_least_squares_solution():
    from optimizer_arena.optimizers import newton

    X, y = _setup()
    theta, history = newton(X, y)
    exact = np.linalg.solve(X.T @ X, X.T @ y)
    assert np.allclose(theta, exact)
    assert len(history) <= 3