import numpy as np

from optimizer_arena.data import (
    add_bias_column,
    load_california_housing,
    make_linear_data,
    standardize,
)
from optimizer_arena.models import hessian
from optimizer_arena.optimizers import gradient_descent, newton
from optimizer_arena.plotting import plot_contour_paths, plot_loss_curves

FIG_DIR = "results/figures"


def figure_california():
    features, y = load_california_housing()
    histories, labels = [], []
    for name, F in [("raw", features), ("standardized", standardize(features))]:
        X = add_bias_column(F)
        lr = 1.0 / np.linalg.eigvalsh(hessian(X)).max()
        _, hist_gd = gradient_descent(X, y, lr=lr, n_iters=1000)
        _, hist_nt = newton(X, y)
        histories += [hist_gd, hist_nt]
        labels += [f"GD ({name})", f"Newton ({name})"]
    plot_loss_curves(
        histories, labels,
        "California Housing: GD vs Newton",
        f"{FIG_DIR}/california_raw_vs_scaled.png",
    )


def figure_contour():
    x, y = make_linear_data()
    X = add_bias_column(x)
    exact = np.linalg.solve(X.T @ X, X.T @ y)
    start = np.array([-4.0, -1.0])
    _, _, path_gd = gradient_descent(X, y, lr=0.025, n_iters=80, theta0=start, return_path=True)
    _, _, path_nt = newton(X, y, theta0=start, return_path=True)
    plot_contour_paths(
        X, y, [path_gd, path_nt], ["Gradient descent", "Newton"], exact,
        "Loss surface: GD path vs Newton jump",
        f"{FIG_DIR}/contour_gd_vs_newton.png",
        w_range=(-2, 12),
    )


def figure_learning_rate():
    x, y = make_linear_data()
    X = add_bias_column(x)
    limit = 2 / np.linalg.eigvalsh(hessian(X)).max()
    print("Stability limit 2 / lambda_max:", limit)
    lrs = [0.0005, 0.01, 0.035]
    labels = ["too small (0.0005)", "good (0.01)", "too large (0.035)"]
    histories = [gradient_descent(X, y, lr=lr, n_iters=200)[1] for lr in lrs]
    plot_loss_curves(
        histories, labels,
        "Learning rate experiment (GD)",
        f"{FIG_DIR}/learning_rate.png",
    )


if __name__ == "__main__":
    figure_california()
    figure_contour()
    figure_learning_rate()