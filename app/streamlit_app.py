"""Optimizer Arena live demo.

Move the learning-rate slider and watch how the gradient descent path changes
on the loss surface, while Newton always jumps straight to the minimum.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from optimizer_arena.data import add_bias_column, make_linear_data  # noqa: E402
from optimizer_arena.models import hessian  # noqa: E402
from optimizer_arena.optimizers import gradient_descent, newton  # noqa: E402


def _finite_prefix(path, limit=1e3):
    """Keep path points until the first one that blows up (for plotting)."""
    ok = np.all(np.isfinite(path), axis=1) & np.all(np.abs(path) < limit, axis=1)
    bad = np.where(~ok)[0]
    return path[: bad[0]] if len(bad) else path


def build_figures(lr, n_iters, start_b, start_w, noise_std, n_samples=100, seed=42):
    """Run GD and Newton on synthetic data and return figures plus summary numbers."""
    x, y = make_linear_data(n_samples=n_samples, noise_std=noise_std, seed=seed)
    X = add_bias_column(x)
    exact = np.linalg.solve(X.T @ X, X.T @ y)

    eigvals = np.linalg.eigvalsh(hessian(X))
    limit = 2.0 / eigvals.max()
    start = np.array([start_b, start_w], dtype=float)

    with np.errstate(all="ignore"):
        theta_gd, hist_gd, path_gd = gradient_descent(
            X, y, lr=lr, n_iters=n_iters, theta0=start, return_path=True
        )
        theta_nt, hist_nt, path_nt = newton(X, y, theta0=start, return_path=True)

        # Loss surface on a (b, w) grid. With X = [1, x]: prediction = b + w * x.
        pad_b = max(2.0, abs(start_b - exact[0]) * 0.3)
        pad_w = max(1.0, abs(start_w - exact[1]) * 0.3)
        bs = np.linspace(min(start_b, exact[0]) - pad_b, max(start_b, exact[0]) + pad_b, 120)
        ws = np.linspace(min(start_w, exact[1]) - pad_w, max(start_w, exact[1]) + pad_w, 120)
        B, W = np.meshgrid(bs, ws)
        pred = B[..., None] + W[..., None] * x
        Z = np.mean((pred - y) ** 2, axis=-1)

    fig_c, ax = plt.subplots(figsize=(6.5, 5.5))
    levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 25)
    ax.contour(B, W, Z, levels=levels, cmap="viridis")
    pg = _finite_prefix(path_gd)
    if len(pg):
        ax.plot(pg[:, 0], pg[:, 1], "o-", ms=3, color="#1f77b4", label="Gradient descent")
    ax.plot(path_nt[:, 0], path_nt[:, 1], "o-", ms=4, color="#d9822b", label="Newton")
    ax.plot(exact[0], exact[1], "k*", ms=15, label="Exact minimum")
    ax.set_xlim(bs[0], bs[-1])
    ax.set_ylim(ws[0], ws[-1])
    ax.set_xlabel("b (intercept)")
    ax.set_ylabel("w (slope)")
    ax.set_title("Loss surface and optimizer paths")
    ax.legend(loc="lower right")

    fig_l, axl = plt.subplots(figsize=(6.5, 5.5))
    hg = np.array(hist_gd, dtype=float)
    hg[~np.isfinite(hg)] = np.nan
    axl.plot(range(len(hg)), hg, color="#1f77b4", label="Gradient descent")
    axl.plot(range(len(hist_nt)), hist_nt, "o-", color="#d9822b", label="Newton")
    axl.set_yscale("log")
    axl.set_xlabel("Iteration")
    axl.set_ylabel("Loss (MSE, log scale)")
    axl.set_title("Loss curves")
    axl.grid(True, alpha=0.3)
    axl.legend()

    final_loss = hist_gd[-1]
    summary = {
        "limit": limit,
        "cond": eigvals.max() / eigvals.min(),
        "exact": exact,
        "theta_gd": theta_gd,
        "theta_newton": theta_nt,
        "final_loss_gd": final_loss,
        "diverged": (not np.isfinite(final_loss)) or final_loss > hist_gd[0],
        "newton_steps": len(hist_nt) - 1,
    }
    return fig_c, fig_l, summary


def main():
    import streamlit as st

    st.set_page_config(page_title="Optimizer Arena", layout="wide")
    st.title("Optimizer Arena: Gradient Descent vs Newton")
    st.caption(
        "Linear regression on synthetic data (true b = 2, w = 3). "
        "GD update: θ ← θ − α∇L.  Newton update: θ ← θ − H⁻¹∇L."
    )

    st.sidebar.header("Controls")
    lr = st.sidebar.slider("Learning rate α", 0.0005, 0.0400, 0.0100, step=0.0005, format="%.4f")
    n_iters = st.sidebar.slider("GD iterations", 10, 1000, 200, step=10)
    start_b = st.sidebar.slider("Start b", -8.0, 8.0, -4.0, step=0.5)
    start_w = st.sidebar.slider("Start w", -4.0, 8.0, -1.0, step=0.5)
    noise_std = st.sidebar.slider("Noise std", 0.0, 3.0, 1.0, step=0.25)
    seed = int(st.sidebar.number_input("Seed", min_value=0, value=42, step=1))

    fig_c, fig_l, s = build_figures(lr, n_iters, start_b, start_w, noise_std, seed=seed)

    c1, c2, c3 = st.columns(3)
    c1.metric("Stability limit 2/λmax", f"{s['limit']:.4f}")
    c2.metric("Condition number of H", f"{s['cond']:.0f}")
    c3.metric("Newton steps", s["newton_steps"])

    if s["diverged"]:
        st.error(f"α = {lr:.4f} is above the stability limit {s['limit']:.4f}: the loss blows up.")
    else:
        st.success(f"α = {lr:.4f} is below the stability limit {s['limit']:.4f}: GD converges.")

    left, right = st.columns(2)
    with left:
        st.pyplot(fig_c)
    with right:
        st.pyplot(fig_l)
    plt.close(fig_c)
    plt.close(fig_l)

    st.subheader("Final parameters (b, w)")
    st.write(
        {
            "exact (least squares)": np.round(s["exact"], 5).tolist(),
            "Newton": np.round(s["theta_newton"], 5).tolist(),
            "Gradient descent": np.round(s["theta_gd"], 5).tolist(),
        }
    )


if __name__ == "__main__":
    main()
