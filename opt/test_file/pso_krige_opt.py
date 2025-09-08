import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from base_krige_opt import Base_Krige_Optimizer

plt.rcParams['font.sans-serif'] = ['SimHei']

class PSO_Krige_Optimizer(Base_Krige_Optimizer):
    def __init__(self, iters=500, particles=30, w=0.7, c1=1.5, c2=1.5):
        super().__init__()
        self.iters = iters
        self.particles = particles
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def particle_swarm_optimize(self, x_train, y_train, z_train, x_test, y_test, z_test,
                               nuggets_range, ranges_range, sills_range):
        lb = np.array([nuggets_range[0], ranges_range[0], sills_range[0]])
        ub = np.array([nuggets_range[-1], ranges_range[-1], sills_range[-1]])
        pos = np.random.uniform(lb, ub, (self.particles, 3))
        vel = np.zeros_like(pos)
        pbest = pos.copy()
        pbest_score = np.array([
            self.evaluate_fitness(p[0], p[1], p[2], x_train, y_train, z_train, x_test, y_test, z_test)
            for p in pos
        ])
        gbest_idx = np.argmin(pbest_score)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_score[gbest_idx]
        for iter in range(self.iters):
            r1 = np.random.rand(self.particles, 3)
            r2 = np.random.rand(self.particles, 3)
            vel = (self.w * vel +
                   self.c1 * r1 * (pbest - pos) +
                   self.c2 * r2 * (gbest - pos))
            pos += vel
            pos = np.clip(pos, lb, ub)
            for i in range(self.particles):
                score = self.evaluate_fitness(pos[i,0], pos[i,1], pos[i,2],
                                              x_train, y_train, z_train, x_test, y_test, z_test)
                if score < pbest_score[i]:
                    pbest[i] = pos[i]
                    pbest_score[i] = score
                    if score < gbest_score:
                        gbest = pos[i].copy()
                        gbest_score = score
        self.nuggets, self.ranges, self.sills = gbest
        return gbest_score, {
            'nugget': gbest[0],
            'range': gbest[1],
            'sill': gbest[2]
        }

    def interpolate_and_compare(self, x, y, z, optimized_params, default_params=None, grid_res=100, target_layer=""):
        grid_x = np.linspace(x.min(), x.max(), grid_res)
        grid_y = np.linspace(y.min(), y.max(), grid_res)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        ok_default = OrdinaryKriging(
            x, y, z,
            variogram_model="spherical",
            # variogram_parameters=default_params,
            verbose=True,
            enable_plotting=False
        )
        z_default, ss_default = ok_default.execute("grid", grid_x, grid_y)
        fitted_params = ok_default.variogram_model_parameters
        print("自动拟合参数（nugget, range, sill）:", fitted_params)
        ok_opt = OrdinaryKriging(
            x, y, z,
            variogram_model="spherical",
            variogram_parameters=optimized_params,
            verbose=True,
            enable_plotting=False
        )
        z_opt, ss_opt = ok_opt.execute("grid", grid_x, grid_y)
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        cs1 = axs[0, 0].contourf(grid_xx, grid_yy, z_default, cmap='viridis', levels=50)
        axs[0, 0].scatter(x, y, c=z, edgecolor='k', s=40)
        axs[0, 0].set_title("Default Parameters")
        axs[0, 0].set_xlabel("X")
        axs[0, 0].set_ylabel("Y")
        fig.colorbar(cs1, ax=axs[0, 0])
        cs2 = axs[0, 1].contourf(grid_xx, grid_yy, z_opt, cmap='viridis', levels=50)
        axs[0, 1].scatter(x, y, c=z, edgecolor='k', s=40)
        axs[0, 1].set_title("Optimized by PSO")
        axs[0, 1].set_xlabel("X")
        axs[0, 1].set_ylabel("Y")
        fig.colorbar(cs2, ax=axs[0, 1])
        extent = (min(grid_x), max(grid_x), min(grid_y), max(grid_y))
        vmin = min(np.min(ss_default), np.min(ss_opt))
        vmax = max(np.max(ss_default), np.max(ss_opt))
        im0 = axs[1, 0].imshow(ss_default, origin='lower', extent=extent, cmap='inferno',
                               vmin=vmin, vmax=vmax)
        axs[1, 0].scatter(x, y, c='white', s=10)
        axs[1, 0].set_title("Prediction Variance (Original)")
        fig.colorbar(im0, ax=axs[1, 0], fraction=0.046, pad=0.04)
        im1 = axs[1, 1].imshow(ss_opt, origin='lower', extent=extent, cmap='inferno',
                               vmin=vmin, vmax=vmax)
        axs[1, 1].scatter(x, y, c='white', s=10)
        axs[1, 1].set_title("Prediction Variance (Optimized)")
        fig.colorbar(im1, ax=axs[1, 1], fraction=0.046, pad=0.04)
        plt.suptitle(f"{target_layer} Kriging Interpolation & Variance Comparison", fontsize=18)
        plt.show()

    def run(self, data_path, target_layer="松散层"):
        (x_train, y_train, z_train), (x_test, y_test, z_test) = self.generate_data(data_path, target_layer=target_layer)
        nuggets_range, ranges_range, sills_range = self.define_parameter_space(x_train, y_train, z_train)
        best_score, best_params = self.particle_swarm_optimize(
            x_train, y_train, z_train,
            x_test, y_test, z_test,
            nuggets_range, ranges_range, sills_range
        )
        print("PSO最优参数:", best_params)

        self.interpolate_and_compare(x_train, y_train, z_train, best_params, target_layer=target_layer)

if __name__ == "__main__":
    target_layer = "松散层"
    data_path = "./data/real_data/地层坐标.xlsx"
    optimizer = PSO_Krige_Optimizer(iters=500, particles=30, w=0.7, c1=1.5, c2=1.5)
    optimizer.run(data_path, target_layer)