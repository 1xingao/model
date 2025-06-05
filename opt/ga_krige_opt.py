import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import pandas as pd
from .base_krige_opt import Base_Krige_Optimizer

plt.rcParams['font.sans-serif'] = ['SimHei']

class GA_Krige_Optimizer(Base_Krige_Optimizer):
    def __init__(self, generations=500, pop_size=50, crossover_rate=0.8, mutation_rate=0.2):
        super().__init__()
        self.generations = generations
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate


    def initialize_population(self, nuggets_range, ranges_range, sills_range):
        pop = []
        for _ in range(self.pop_size):
            nugget = np.random.uniform(nuggets_range[0], nuggets_range[-1])
            range_ = np.random.uniform(ranges_range[0], ranges_range[-1])
            sill = np.random.uniform(sills_range[0], sills_range[-1])
            pop.append([nugget, range_, sill])
        return np.array(pop)

    def select(self, population, fitness):
        idx = np.argsort(fitness)
        return population[idx[:self.pop_size // 2]]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, 3)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual, nuggets_range, ranges_range, sills_range):
        for i in range(3):
            if np.random.rand() < self.mutation_rate:
                if i == 0:
                    individual[i] = np.random.uniform(nuggets_range[0], nuggets_range[-1])
                elif i == 1:
                    individual[i] = np.random.uniform(ranges_range[0], ranges_range[-1])
                else:
                    individual[i] = np.random.uniform(sills_range[0], sills_range[-1])
        return individual

    def genetic_optimize(self, x_train, y_train, z_train, x_test, y_test, z_test,
                         nuggets_range, ranges_range, sills_range):
        population = self.initialize_population(nuggets_range, ranges_range, sills_range)
        best_score = float('inf')
        best_params = None
        for gen in range(self.generations):
            fitness = np.array([
                self.evaluate_fitness(ind[0], ind[1], ind[2],
                                     x_train, y_train, z_train,
                                     x_test, y_test, z_test)
                for ind in population
            ])
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_score:
                best_score = fitness[min_idx]
                best_params = population[min_idx].copy()
            selected = self.select(population, fitness)
            next_population = []
            while len(next_population) < self.pop_size:
                parents = selected[np.random.choice(len(selected), 2, replace=False)]
                child1, child2 = self.crossover(parents[0], parents[1])
                child1 = self.mutate(child1, nuggets_range, ranges_range, sills_range)
                child2 = self.mutate(child2, nuggets_range, ranges_range, sills_range)
                next_population.extend([child1, child2])
            population = np.array(next_population[:self.pop_size])
        self.nuggets, self.ranges, self.sills = best_params
        return best_score, {
            'nugget': self.nuggets,
            'range': self.ranges,
            'sill': self.sills
        }

    def interpolate_and_compare(self, x, y, z, optimized_params, default_params=None, grid_res=100, target_layer="黄土"):
        grid_x = np.linspace(x.min(), x.max(), grid_res)
        grid_y = np.linspace(y.min(), y.max(), grid_res)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        ok_default = OrdinaryKriging(
            x, y, z,
            variogram_model="spherical",
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
        axs[0, 1].set_title("Optimized by GA")
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

    def run(self, data_path, target_layer="黄土"):
        (x_train, y_train, z_train), (x_test, y_test, z_test) = self.generate_data(data_path, target_layer=target_layer)
        nuggets_range, ranges_range, sills_range = self.define_parameter_space(x_train, y_train, z_train)
        best_score, best_params = self.genetic_optimize(
            x_train, y_train, z_train,
            x_test, y_test, z_test,
            nuggets_range, ranges_range, sills_range
        )
        print("GA最优参数:", best_params)
        df = pd.read_excel(data_path)
        layer_df = df[df["地层"] == target_layer]
        x = layer_df["X"].values.astype(np.float64)
        y = layer_df["Y"].values.astype(np.float64)
        z = layer_df["厚度"].values.astype(np.float64)
        self.interpolate_and_compare(x, y, z, best_params, target_layer=target_layer)

if __name__ == "__main__":
    target_layer = "填土"
    data_path = "./data/增强后的钻孔数据.xlsx"
    optimizer = GA_Krige_Optimizer(generations=100, pop_size=30, crossover_rate=0.8, mutation_rate=0.2)
    optimizer.run(data_path, target_layer)