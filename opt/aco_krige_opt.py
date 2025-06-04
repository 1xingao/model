import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

plt.rcParams['font.sans-serif'] = ['SimHei']
class ACO_Krige_Optimizer:
    def __init__(self, iters=500, ants=100, decay=0.8):
        self.iters = iters
        self.ants = ants
        self.decay = decay
        self.nuggets = None
        self.ranges = None
        self.sills = None

    def generate_data(self, data_path, target_layer="黄土", seed=17,  train_ratio=0.7):
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
        sills_range   = np.linspace(0.5 * np.var(z), 3.0 * np.var(z), 10)
        return nuggets_range, ranges_range, sills_range

    # def auto_define_parameter_space(self, x, y, z, levels=5):
    #     var_z = np.var(z)
    #     nuggets_range = np.linspace(0, var_z * 0.5, levels)
    #     coords = np.vstack([x, y]).T
    #     dists = pdist(coords)
    #     range_min = np.percentile(dists, 5)
    #     range_max = np.percentile(dists, 95)
    #     ranges_range = np.linspace(range_min, range_max, levels)
    #     sills_range = np.linspace(var_z * 0.5, var_z * 2.0, levels)
    #     return nuggets_range, ranges_range, sills_range
    def auto_define_parameter_space(self, x, y, z, levels=5):
        var_z = np.var(z)
        # 计算区域最大距离（约束变程上限）
        domain_size = np.sqrt((x.max()-x.min())**2 + (y.max()-y.min())**2)
        
        nuggets_range = [0, var_z * 0.5]
        coords = np.vstack([x, y]).T
        dists = pdist(coords)
        range_min = np.percentile(dists, 5)
        range_max = min(np.percentile(dists, 95), domain_size*1.2)  # 关键约束
        
        ranges_range = [range_min, range_max]
        sills_range = [max(var_z*0.3, 0.1), var_z*2.0]  # 添加最小值约束
        return nuggets_range, ranges_range, sills_range
    

    def get_parameter(self):
        return self.nuggets, self.ranges, self.sills

    def evaluate_fitness(self, nugget, range_, sill, x_train, y_train, z_train, x_test, y_test, z_test):
        # # 参数有效性检查（核心修复）
        # domain_size = np.sqrt((x_train.max()-x_train.min())**2 + (y_train.max()-y_train.min())**2)
        # if range_ > domain_size * 1.5:  # 变程过大
        #     return 1e6
        # if sill < nugget + 1e-6:  # 基台值过小
        #     return 1e6
        # if abs(sill - nugget) < 1e-6:  # 部分基台值无效
        #     return 1e6
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
            # return mse
            trend_corr, _ = spearmanr(z_pred, z_test)
            penalty = (1 - trend_corr)**2  # 趋势越不一致，惩罚越大

            return mse * (1 + penalty)  # 趋势反转会导致惩罚倍增
        except Exception:
            return 1e6

    def ant_colony_optimize(self, x_train, y_train, z_train, x_test, y_test, z_test,
                            nuggets, ranges, sills, alpha=1.1, beta = 1.5):
        
        pheromone = np.ones((len(nuggets), len(ranges), len(sills)))
        best_score = float('inf')
        best_indices = (0, 0, 0)

        # 预先计算启发因子（这里用参数空间均匀分布，启发因子可设为1）
        heuristic = np.ones_like(pheromone)

        for iteration in range(self.iters):
            scores = []
            paths = []

            for ant in range(self.ants):
                # 计算每个参数的选择概率
                prob_nugget = (pheromone.sum(axis=(1,2)) ** alpha) * (heuristic.sum(axis=(1,2)) ** beta)
                prob_nugget /= prob_nugget.sum()
                i = np.random.choice(len(nuggets), p=prob_nugget)

                prob_range = (pheromone[i].sum(axis=1) ** alpha) * (heuristic[i].sum(axis=1) ** beta)
                prob_range /= prob_range.sum()
                j = np.random.choice(len(ranges), p=prob_range)

                prob_sill = (pheromone[i, j] ** alpha) * (heuristic[i, j] ** beta)
                prob_sill /= prob_sill.sum()
                k = np.random.choice(len(sills), p=prob_sill)

                # 评价当前参数组合
                score = self.evaluate_fitness(
                    nuggets[i], ranges[j], sills[k],
                    x_train, y_train, z_train,
                    x_test, y_test, z_test
                )
                scores.append(score)
                paths.append((i, j, k))

                # 更新全局最优
                if score < best_score:
                    best_score = score
                    best_indices = (i, j, k)

            # 信息素挥发
            pheromone *= self.decay

            # 信息素增强（分数越低，信息素增量越大）
            for (i, j, k), score in zip(paths, scores):
                pheromone[i, j, k] += 1.0 / (score + 1e-6)

        i, j, k = best_indices
        self.nuggets = nuggets[i]
        self.ranges = ranges[j]
        self.sills = sills[k]
        return best_score, {
            'nugget': nuggets[i],
            'range': ranges[j],
            'sill': sills[k]
        }


    def interpolate_and_compare(self, x, y, z, optimized_params, default_params=None, grid_res=100):
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
        axs[0, 1].set_title("Optimized by ACO")
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
        plt.suptitle(f"{target_layer}Kriging Interpolation & Variance Comparison", fontsize=18)
        # plt.savefig(f"./pic/{target_layer}_krige_aco.png", dpi=300)
        plt.show()

    def run(self, data_path, target_layer="黄土"):
        (x_train, y_train, z_train), (x_test, y_test, z_test) = self.generate_data(data_path, target_layer=target_layer)
        # 使用自适应参数空间
        nuggets_range, ranges_range, sills_range = self.define_parameter_space(x_train, y_train, z_train)

        best_score, best_params = self.ant_colony_optimize(
            x_train, y_train, z_train,
            x_test, y_test, z_test,
            nuggets_range, ranges_range, sills_range,
            alpha=1.0, beta=2.0
        )
        print("ACO最优参数:", best_params)
        df = pd.read_excel(data_path)
        layer_df = df[df["地层"] == target_layer]
        x = layer_df["X"].values.astype(np.float64)
        y = layer_df["Y"].values.astype(np.float64)
        z = layer_df["厚度"].values.astype(np.float64)
        self.interpolate_and_compare(x, y, z, best_params)

if __name__ == "__main__":
    target_layer = "填土"
    data_path = "./data/增强后的钻孔数据.xlsx"
    # 增加蚂蚁数量和迭代次数
    optimizer = ACO_Krige_Optimizer(iters=500, ants=30, decay=0.8)
    optimizer.run(data_path, target_layer)
