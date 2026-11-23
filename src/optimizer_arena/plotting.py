import os

import matplotlib.pyplot as plt
import numpy as np

from optimizer_arena.models import mse_loss


def plot_loss_curves(histories, labels, title, path):
    """Plot several loss histories on one log-scale chart and save it."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for history, label in zip(histories, labels):
        ax.plot(range(len(history)), history, marker="o" if len(history) <= 5 else None, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (MSE, log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_contour_paths(X, y, paths, labels, optimum, title, path,
                       b_range=(-5, 5), w_range=(-2, 6)):
    """Draw loss contours over (b, w) and overlay optimizer paths."""
    bs = np.linspace(b_range[0], b_range[1], 120)
    ws = np.linspace(w_range[0], w_range[1], 120)
    B, W = np.meshgrid(bs, ws)
    Z = np.zeros_like(B)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            Z[i, j] = mse_loss(X, y, np.array([B[i, j], W[i, j]]))

    fig, ax = plt.subplots(figsize=(7, 6))
    levels = np.logspace(np.log10(Z.min()), np.log10(Z.max()), 25)
    ax.contour(B, W, Z, levels=levels, cmap="viridis")
    for p, label in zip(paths, labels):
        ax.plot(p[:, 0], p[:, 1], "o-", markersize=3, label=label)
    ax.plot(optimum[0], optimum[1], "k*", markersize=15, label="Exact minimum")
    ax.set_xlim(b_range)
    ax.set_ylim(w_range)
    ax.set_xlabel("b (intercept)")
    ax.set_ylabel("w (slope)")
    ax.set_title(title)
    ax.legend()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)