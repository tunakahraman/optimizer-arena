import numpy as np

from optimizer_arena.data import add_bias_column, make_linear_data
from optimizer_arena.models import hessian
from optimizer_arena.optimizers import gradient_descent, newton

x, y = make_linear_data()
X = add_bias_column(x)
exact = np.linalg.solve(X.T @ X, X.T @ y)

theta_gd, hist_gd = gradient_descent(X, y, lr=0.01, n_iters=1000)
theta_nt, hist_nt = newton(X, y)

print("Exact   :", exact)
print("GD      :", theta_gd, " steps:", len(hist_gd) - 1, " final loss:", hist_gd[-1])
print("Newton  :", theta_nt, " steps:", len(hist_nt) - 1, " final loss:", hist_nt[-1])
print("Condition number of H:", np.linalg.cond(hessian(X)))