"""Compare our NumPy optimizers with PyTorch on California Housing.

What this script checks:
1. Autograd (PyTorch) gradient and Hessian match our hand-derived formulas.
2. torch.optim.SGD (full batch) reproduces our gradient_descent step by step.
3. How torch.optim.Adam and torch.optim.LBFGS behave on the same problem.

Run from the project root:
    python experiments/compare_pytorch.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from optimizer_arena.data import add_bias_column, load_california_housing, standardize  # noqa: E402
from optimizer_arena.models import gradient, hessian  # noqa: E402
from optimizer_arena.optimizers import gradient_descent, newton  # noqa: E402

N_ITERS = 1000


def mse(Xt, yt, theta):
    return ((Xt @ theta - yt) ** 2).mean()


def check_autograd(X, y):
    """Compare our analytic gradient/Hessian with PyTorch autograd at a random point."""
    rng = np.random.default_rng(0)
    theta0 = rng.normal(size=X.shape[1])
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    theta = torch.tensor(theta0, requires_grad=True)
    mse(Xt, yt, theta).backward()
    grad_err = np.abs(theta.grad.numpy() - gradient(X, y, theta0)).max()

    H_auto = torch.autograd.functional.hessian(
        lambda t: mse(Xt, yt, t), torch.tensor(theta0)
    ).numpy()
    hess_err = np.abs(H_auto - hessian(X)).max()
    return grad_err, hess_err


def run_first_order(make_optimizer, X, y, n_iters):
    """Train theta with a torch optimizer. history[k] is the loss after k updates."""
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    theta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = make_optimizer([theta])
    history = []
    for _ in range(n_iters):
        opt.zero_grad()
        loss = mse(Xt, yt, theta)
        history.append(loss.item())
        loss.backward()
        opt.step()
    with torch.no_grad():
        history.append(mse(Xt, yt, theta).item())
    return theta.detach().numpy().copy(), history


def run_lbfgs(X, y, n_outer=10):
    """torch.optim.LBFGS: a quasi-Newton method (approximates the Hessian)."""
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    theta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [theta], lr=1.0, max_iter=20, history_size=10, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = mse(Xt, yt, theta)
        loss.backward()
        return loss

    history = [mse(Xt, yt, theta).item()]
    for _ in range(n_outer):
        opt.step(closure)
        with torch.no_grad():
            history.append(mse(Xt, yt, theta).item())
    return theta.detach().numpy().copy(), history


def main():
    features, y = load_california_housing()
    X = add_bias_column(standardize(features))  # float64, shape (20640, 9)
    exact = np.linalg.solve(X.T @ X, X.T @ y)
    lr = 1.0 / np.linalg.eigvalsh(hessian(X)).max()

    grad_err, hess_err = check_autograd(X, y)
    print("Autograd check (max abs difference to our formulas)")
    print(f"  gradient: {grad_err:.2e}")
    print(f"  Hessian : {hess_err:.2e}\n")

    runs = {}
    runs["Ours: GD"] = gradient_descent(X, y, lr=lr, n_iters=N_ITERS)
    runs["Ours: Newton"] = newton(X, y)
    runs["PyTorch: SGD (full batch)"] = run_first_order(
        lambda p: torch.optim.SGD(p, lr=lr), X, y, N_ITERS
    )
    runs["PyTorch: Adam (lr=0.05)"] = run_first_order(
        lambda p: torch.optim.Adam(p, lr=0.05), X, y, N_ITERS
    )
    runs["PyTorch: LBFGS"] = run_lbfgs(X, y)

    ours_gd = runs["Ours: GD"][0]
    torch_sgd = runs["PyTorch: SGD (full batch)"][0]
    print(f"Learning rate used for GD / SGD: {lr:.4f}")
    print(f"max |theta_ours_GD - theta_torch_SGD| = {np.abs(ours_gd - torch_sgd).max():.2e}\n")

    print(f"{'Method':<28}{'updates':>9}{'final loss':>14}{'max|theta-exact|':>20}")
    for name, (theta, hist) in runs.items():
        print(f"{name:<28}{len(hist) - 1:>9}{hist[-1]:>14.6f}{np.abs(theta - exact).max():>20.2e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "Ours: GD": dict(color="#1f77b4", lw=3, alpha=0.5),
        "PyTorch: SGD (full batch)": dict(color="#1f77b4", ls="--", lw=1.5),
        "PyTorch: Adam (lr=0.05)": dict(color="#2ca02c"),
        "Ours: Newton": dict(color="#d9822b", marker="o"),
        "PyTorch: LBFGS": dict(color="#c8553d", marker="s", ls=":"),
    }
    for name, (_, hist) in runs.items():
        ax.plot(range(len(hist)), hist, label=name, **styles[name])
    ax.set_yscale("log")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss (MSE, log scale)")
    ax.set_title("California Housing: our optimizers vs PyTorch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    out = os.path.join(fig_dir, "pytorch_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")

    ax.set_xlim(-2, 100)
    out_zoom = os.path.join(fig_dir, "pytorch_comparison_zoom.png")
    fig.savefig(out_zoom, dpi=150, bbox_inches="tight")

    print(f"\nSaved figures to {out} and {out_zoom}")


if __name__ == "__main__":
    main()