import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

class Default_Krige:
    def __init__(self,model='spherical'):
        self.model = model
        self.nugget = None
        self.range_ = None
        self.sill = None
        self.ok = None
        self.fitted_params = None

    def fit_parameter(self, x, y, z, nugget=0.1, range_=1.0, sill=1.0):
        self.nugget = nugget
        self.range_ = range_
        self.sill = sill
        self.ok = OrdinaryKriging(
            x, y, z,
            variogram_model=self.model,
            variogram_parameters=[self.nugget, self.range_, self.sill],
            verbose=False,
            enable_plotting=False
        )

    def get_parameter(self):
        if self.ok is not None:
            return self.fitted_params
        else:
            raise ValueError("Kriging model has not been fitted yet.")
        
    def fit_default(self, x, y, z):
        grid_res = 100
        grid_x = np.linspace(x.min(), x.max(), grid_res)
        grid_y = np.linspace(y.min(), y.max(), grid_res)
        self.ok = OrdinaryKriging(
            x, y, z,
            variogram_model=self.model,
            verbose=False,
            enable_plotting=False
        )
        z_default, ss_default = self.ok.execute("grid", grid_x, grid_y)
        fitted_params = self.ok.variogram_model_parameters
        self.fitted_params = fitted_params

    def predict(self, grid_x, grid_y):
        z_pre,ss_pre = self.ok.execute('grid', grid_x, grid_y)
        return z_pre, ss_pre