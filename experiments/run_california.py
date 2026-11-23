import numpy as np

from optimizer_arena.data import add_bias_column, load_california_housing, standardize
from optimizer_arena.models import hessian
from optimizer_arena.optimizers import gradient_descent, newton

features, y = load_california_housing()

for name, F in [("raw", features), ("standardized", standardize(features))]:
    X = add_bias_column(F)
    H = hessian(X)
    lam_max = np.linalg.eigvalsh(H).max()
    lr = 1.0 / lam_max

    exact = np.linalg.solve(X.T @ X, X.T @ y)
    theta_gd, hist_gd = gradient_descent(X, y, lr=lr, n_iters=1000)
    theta_nt, hist_nt = newton(X, y)

    print(f"--- {name} ---")
    print("cond(H):", np.linalg.cond(H), " lr:", lr)
    print("GD loss after 1000 steps:", hist_gd[-1])
    print("Newton loss after", len(hist_nt) - 1, "step(s):", hist_nt[-1])
    print("Exact loss:", np.mean((X @ exact - y) ** 2))