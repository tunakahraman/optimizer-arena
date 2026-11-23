import numpy as np

def make_linear_data(n_samples=100, true_w=3.0, true_b=2.0, noise_std=1.0, seed=42):
    """Return x and y where y= true_w * x + true_b + Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, size=n_samples)
    noise = rng.normal(0, noise_std, size=n_samples)
    y = true_w * x + true_b + noise
    return x, y
    
def add_bias_column(x):
    """Return a design matrix whose first column is all ones."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ones = np.ones((x.shape[0], 1))
    return np.hstack((ones, x))

def load_california_housing():
    """Return raw feature matrix (no bias column) and target (units of $100k)."""
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    return data.data, data.target

def standardize(features):
    """Return features scaled to mean 0 and standard deviation 1 per column."""
    return (features - features.mean(axis=0)) / features.std(axis=0)