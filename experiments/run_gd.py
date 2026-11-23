import numpy as np

from optimizer_arena.data import add_bias_column, make_linear_data
from optimizer_arena.optimizers import gradient_descent

x, y = make_linear_data()
X = add_bias_column(x)

theta, history = gradient_descent(X, y, lr=0.01, n_iters=1000)
exact = np.linalg.solve(X.T @ X, X.T @ y)

print("GD theta (b, w):", theta)
print("Exact least squares:", exact)
print("Loss: first =", history[0], " last =", history[-1])