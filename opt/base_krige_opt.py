import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

class Base_Krige_Optimizer:
    def __init__(self):
        self.nuggets = None
        self.ranges = None
        self.sills = None

    def generate_data(self, data_path, target_layer="黄土", seed=0, train_ratio=0.7):
        np.random.seed(seed)
        df = pd.read_excel(data_path)
        layer_df = df[df["地层"] == target_layer]
        x = layer_df["X"].values.astype(np.float64)
        y = layer_df["Y"].values.astype(np.float64)
        z = layer_df["厚度"].values.astype(np.float64)
        n_points = len(x)
        n_train = int(n_points * train_ratio)
        train_idx = np.random.choice(n_points, n_train, replace=False)
        test_idx = np.setdiff1d(np.arange(n_points), train_idx)
        return (x[train_idx], y[train_idx], z[train_idx]), (x[test_idx], y[test_idx], z[test_idx])

    def define_parameter_space(self, x, y, z):
        domain_size = max(x) - min(x)
        nuggets_range = np.linspace(0, np.var(z) * 0.8, 10)
        ranges_range = np.linspace(0.05 * domain_size, 2.0 * domain_size, 10)
        sills_range = np.linspace(0.5 * np.var(z), 3.0 * np.var(z), 10)
        return nuggets_range, ranges_range, sills_range

    def get_parameter(self):
        return self.nuggets, self.ranges, self.sills

    def evaluate_fitness(self, nugget, range_, sill, x_train, y_train, z_train, x_test, y_test, z_test):
        try:
            ok = OrdinaryKriging(
                x_train, y_train, z_train,
                variogram_model='spherical',
                variogram_parameters=[nugget, range_, sill],
                verbose=False,
                enable_plotting=False
            )
            z_pred, _ = ok.execute('points', x_test, y_test)
            mse = np.mean((z_test - z_pred) ** 2)
            trend_corr, _ = spearmanr(z_pred, z_test)
            penalty = (1 - trend_corr) ** 2
            return mse * (1 + penalty)
        except Exception:
            return 1e6
