import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

def generate_data(target_layer="黄土", seed=0, n_points=30, train_ratio=0.7):
    np.random.seed(seed)
    # x = np.random.uniform(0, 100, n_points)
    # y = np.random.uniform(0, 100, n_points)
    # z = np.sin(x / 10.0) + np.cos(y / 10.0) + np.random.normal(0, 0.1, n_points)
    df = pd.read_excel("./data/增强后的钻孔数据.xlsx")

    # Step 2: 提取目标地层（例如 "黄土"）的坐标和厚度

    layer_df = df[df["地层"] == target_layer]

    # 获取坐标和厚度
    x = layer_df["X"].values.astype(np.float64)
    y = layer_df["Y"].values.astype(np.float64)
    z = layer_df["厚度"].values.astype(np.float64)

    n_train = int(n_points * train_ratio)
    train_idx = np.random.choice(n_points, n_train, replace=False)
    test_idx = np.setdiff1d(np.arange(n_points), train_idx)

    return (x[train_idx], y[train_idx], z[train_idx]), (x[test_idx], y[test_idx], z[test_idx])

def define_parameter_space(x,y,z):
    domain_size = max(x) - min(x)
    nuggets = np.linspace(0, np.var(z) * 0.8, 10)
    ranges = np.linspace(0.05 * domain_size, 2.0 * domain_size, 10)
    sills   = np.linspace(0.5 * np.var(z), 3.0 * np.var(z), 10)
    return nuggets, ranges, sills

def auto_define_parameter_space(x, y, z, levels=5):


    # nugget: 微小变异，一般小
    var_z = np.var(z)
    nuggets = np.linspace(0, var_z * 0.5, levels)

    # range: 空间影响范围
    coords = np.vstack([x, y]).T
    dists = pdist(coords)
    range_min = np.percentile(dists, 5)
    range_max = np.percentile(dists, 95)
    ranges = np.linspace(range_min, range_max, levels)

    # sill: 方差范围
    sills = np.linspace(var_z * 0.5, var_z * 2.0, levels)

    return nuggets, ranges, sills


def evaluate_fitness(nugget, sill, rang, x_train, y_train, z_train, x_test, y_test, z_test):
    try:
        ok = OrdinaryKriging(
            x_train, y_train, z_train,
            variogram_model='spherical',
            variogram_parameters=[nugget, rang, sill],
            verbose=False,
            enable_plotting=False
        )
        z_pred, _ = ok.execute('points', x_test, y_test)
        mse = np.mean((z_test - z_pred)**2)

        # 加入趋势惩罚项
        trend_corr, _ = spearmanr(z_pred, z_test)
        penalty = (1 - trend_corr)**2  # 趋势越不一致，惩罚越大

        return mse * (1 + penalty)  # 趋势反转会导致惩罚倍增

    except Exception:
        return 1e6  # 出错严重惩罚

def ant_colony_optimize(x_train, y_train, z_train, x_test, y_test, z_test,
                        nuggets, ranges, sills, iters=500, ants=100, decay=0.8):
    pheromone = np.ones((len(nuggets), len(ranges), len(sills)))
    best_score = float('inf')
    best_indices = None

    for _ in range(iters):
        scores = []
        paths = []

        for _ in range(ants):
            i = np.random.choice(len(nuggets), p=pheromone.sum((1,2))/pheromone.sum())
            j = np.random.choice(len(ranges), p=pheromone[i].sum(1)/pheromone[i].sum())
            k = np.random.choice(len(sills), p=pheromone[i,j]/pheromone[i,j].sum())

            score = evaluate_fitness(
                x_train, y_train, z_train,
                x_test, y_test, z_test,
                nuggets[i], ranges[j], sills[k]
            )

            scores.append(score)
            paths.append((i, j, k))

        # 信息素更新
        Q = 1.0
        pheromone *= decay
        for (i, j, k), score in zip(paths, scores):
            pheromone[i, j, k] += Q/ (score + 1e-6)

        # 全局最优更新
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_indices = paths[min_idx]

    i, j, k = best_indices
    return best_score, {
        'nugget': nuggets[i],
        'range': ranges[j],
        'sill': sills[k]
    }


def visualize_prediction_variance_comparison(ss_default, ss_opt, x, y, grid_x, grid_y):
    """
    可视化优化前后预测方差，并统一 colorbar 范围，便于比较。
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    extent = (min(grid_x), max(grid_x), min(grid_y), max(grid_y))

    # 统一 colorbar 范围
    vmin = min(np.min(ss_default), np.min(ss_opt))
    vmax = max(np.max(ss_default), np.max(ss_opt))

    im0 = axs[0].imshow(ss_default, origin='lower', extent=extent, cmap='inferno',
                        vmin=vmin, vmax=vmax)
    axs[0].scatter(x, y, c='white', s=10)
    axs[0].set_title("Prediction Variance (Original)")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(ss_opt, origin='lower', extent=extent, cmap='inferno',
                        vmin=vmin, vmax=vmax)
    axs[1].scatter(x, y, c='white', s=10)
    axs[1].set_title("Prediction Variance (Optimized)")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"./pic/{target_layer}_prediction_variance_comparison.png")
    plt.show()
    



def interpolate_and_compare(x, y, z, optimized_params, default_params=None, grid_res=100):
    # 构造网格
    grid_x = np.linspace(x.min(), x.max(), grid_res)
    grid_y = np.linspace(y.min(), y.max(), grid_res)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # # 默认参数（若未提供）
    # if default_params is None:
    #     var_z = np.var(z)
    #     default_params = {
    #         'nugget': 0.1 * var_z,
    #         'range': 0.25 * (x.max() - x.min()),
    #         'sill': var_z
    #     }

    # 插值器：默认参数
    ok_default = OrdinaryKriging(
        x, y, z,
        variogram_model="spherical",
        # variogram_parameters=default_params,
        verbose=True, #自动拟合
        enable_plotting=False
    )
    z_default, ss_default = ok_default.execute("grid", grid_x, grid_y)
    fitted_params = ok_default.variogram_model_parameters
    print("自动拟合参数（nugget, range, sill）:", fitted_params)

    # 插值器：优化参数
    ok_opt = OrdinaryKriging(
        x, y, z,
        variogram_model="spherical",
        variogram_parameters=optimized_params,
        verbose=False,
        enable_plotting=False
    )
    z_opt, ss_opt = ok_opt.execute("grid", grid_x, grid_y)

    # 可视化：对比展示
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    cs1 = axs[0].contourf(grid_xx, grid_yy, z_default, cmap='viridis', levels=50)
    axs[0].scatter(x, y, c=z, edgecolor='k', s=40)
    axs[0].set_title("Default Parameters")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    fig.colorbar(cs1, ax=axs[0])

    cs2 = axs[1].contourf(grid_xx, grid_yy, z_opt, cmap='viridis', levels=50)
    axs[1].scatter(x, y, c=z, edgecolor='k', s=40)
    axs[1].set_title("Optimized by ACO")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    fig.colorbar(cs2, ax=axs[1])

    plt.suptitle("Kriging Interpolation Comparison", fontsize=16)
    plt.savefig(f"./pic/{target_layer}_kriging_interpolation_comparison.png")
    plt.show()
    

    # 可视化预测方差对比
    visualize_prediction_variance_comparison(ss_default, ss_opt, x, y, grid_x, grid_y)


def main(target_layer="黄土"):  
    # 1. 生成模拟钻孔数据
    (x_train, y_train, z_train), (x_test, y_test, z_test) = generate_data(target_layer=target_layer)

    # 2. 定义参数搜索空间
    nuggets, ranges, sills = define_parameter_space(x_train,y_train,z_train)
    # nuggets, ranges, sills = auto_define_parameter_space(x_train, y_train, z_train)


    # 3. 执行蚁群优化
    best_score, best_params = ant_colony_optimize(
        x_train, y_train, z_train,
        x_test, y_test, z_test,
        nuggets, ranges, sills,
        iters=500, ants=20, decay=0.8
    )
    
    # best_params = {'nugget': 1.03738008, 'range': 29.20535359, 'sill': 1.62593025}
    # 4. 输出最优解结果
    print("最优参数:", best_params)

    interpolate_and_compare(x_train, y_train, z_train, best_params)


if __name__ == "__main__":
    target_layer = "砂岩"
    main(target_layer)
