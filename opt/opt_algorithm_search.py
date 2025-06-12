import numpy as np
import matplotlib.pyplot as plt
from pso_krige_opt import PSO_Krige_Optimizer
from aco_krige_opt import ACO_Krige_Optimizer
from ga_krige_opt import GA_Krige_Optimizer

from pykrige.ok import OrdinaryKriging

from scipy.spatial.distance import pdist, squareform

def plot_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters=None, nlags=10, ax=None, label=None):
    """
    绘制实验半方差与理论变差函数曲线
    """
    
    # 1. 计算所有点对的距离和半方差
    coords = np.vstack((x, y)).T
    dists = squareform(pdist(coords))
    semivariances = 0.5 * (z[:, None] - z[None, :]) ** 2
    triu_indices = np.triu_indices_from(dists, k=1)
    h = dists[triu_indices]
    gamma = semivariances[triu_indices]
    # 2. 按距离分组
    bins = np.linspace(h.min(), h.max(), nlags + 1)
    bin_indices = np.digitize(h, bins)
    bin_centers = []
    gamma_means = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if np.any(mask):
            bin_centers.append(h[mask].mean())
            gamma_means.append(gamma[mask].mean())
    bin_centers = np.array(bin_centers)
    gamma_means = np.array(gamma_means)
    # 3. 理论变差函数
    def spherical(h, nugget, rang, sill):
        h = np.array(h)
        y = np.piecewise(
            h,
            [h <= rang, h > rang],
            [lambda x: nugget + (sill - nugget) * (1.5 * x / rang - 0.5 * (x / rang) ** 3), lambda x: sill]
        )
        return y
    if variogram_parameters is not None:
        nugget = variogram_parameters['nugget']
        rang = variogram_parameters['range']
        sill = variogram_parameters['sill']
        h_fit = np.linspace(0, h.max(), 100)
        if variogram_model == 'spherical':
            gamma_fit = spherical(h_fit, nugget, rang, sill)
        else:
            raise NotImplementedError("只实现了spherical模型")
    else:
        h_fit = None
        gamma_fit = None
    # 4. 绘图
    if ax is None:
        ax = plt.gca()
    ax.scatter(bin_centers, gamma_means, color='b', label='实验半方差' if label is None else f'{label} 实验半方差')
    if h_fit is not None:
        ax.plot(h_fit, gamma_fit, color='r', label='理论变差函数' if label is None else f'{label} 理论变差函数')
    ax.set_xlabel('距离 h')
    ax.set_ylabel('半方差 γ(h)')
    ax.set_title('半方差-距离 拟合关系' if label is None else label)
    ax.legend()
    ax.grid(True)


def plot_all_pairs_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters=None, ax=None, label=None):
    """
    直接绘制所有点对的距离和半方差的散点图，并可叠加理论变差函数曲线
    """
    from scipy.spatial.distance import pdist, squareform
    # 1. 计算所有点对的距离和半方差
    coords = np.vstack((x, y)).T
    dists = squareform(pdist(coords))
    semivariances = 0.5 * (z[:, None] - z[None, :]) ** 2
    triu_indices = np.triu_indices_from(dists, k=1)
    h = dists[triu_indices]
    gamma = semivariances[triu_indices]
    # 2. 理论变差函数
    def spherical(h, nugget, rang, sill):
        h = np.array(h)
        y = np.piecewise(
            h,
            [h <= rang, h > rang],
            [lambda x: nugget + (sill - nugget) * (1.5 * x / rang - 0.5 * (x / rang) ** 3), lambda x: sill]
        )
        return y
    if variogram_parameters is not None:
        nugget = variogram_parameters['nugget']
        rang = variogram_parameters['range']
        sill = variogram_parameters['sill']
        h_fit = np.linspace(0, h.max(), 100)
        if variogram_model == 'spherical':
            gamma_fit = spherical(h_fit, nugget, rang, sill)
        else:
            raise NotImplementedError("只实现了spherical模型")
    else:
        h_fit = None
        gamma_fit = None
    # 3. 绘图
    if ax is None:
        ax = plt.gca()
    ax.scatter(h, gamma, color='b', s=10, alpha=0.5, label='所有点对' if label is None else f'{label} 所有点对')
    if h_fit is not None:
        ax.plot(h_fit, gamma_fit, color='r', label='理论变差函数' if label is None else f'{label} 理论变差函数')
    ax.set_xlabel('距离 h')
    ax.set_ylabel('半方差 γ(h)')
    ax.set_title('半方差-距离 全点对关系' if label is None else label)
    ax.legend()
    ax.grid(True)


def visualize_kriging_results(pso_optimizer, aco_optimizer, ga_optimizer, target_layer, train_data):
    # 获取参数
    pso_params = pso_optimizer.get_parameter()
    aco_params = aco_optimizer.get_parameter()
    ga_params = ga_optimizer.get_parameter()

    # 生成网格
    x, y, z = train_data
    grid_res = 100
    grid_x = np.linspace(x.min(), x.max(), grid_res)
    grid_y = np.linspace(y.min(), y.max(), grid_res)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # PSO 克里金插值
    ok_pso = OrdinaryKriging(
        x, y, z,
        variogram_model='spherical',
        variogram_parameters={'nugget': pso_params[0], 'range': pso_params[1], 'sill': pso_params[2]},
        verbose=False,
        enable_plotting=False
    )
    pso_grid, pso_var = ok_pso.execute('grid', grid_x, grid_y)

    # ACO 克里金插值
    ok_aco = OrdinaryKriging(
        x, y, z,
        variogram_model='spherical',
        variogram_parameters={'nugget': aco_params[0], 'range': aco_params[1], 'sill': aco_params[2]},
        verbose=False,
        enable_plotting=False
    )
    aco_grid, aco_var = ok_aco.execute('grid', grid_x, grid_y)

    # GA 克里金插值
    ok_ga = OrdinaryKriging(
        x, y, z,
        variogram_model='spherical',
        variogram_parameters={'nugget': ga_params[0], 'range': ga_params[1], 'sill': ga_params[2]},
        verbose=False,
        enable_plotting=False
    )
    ga_grid, ga_var = ok_ga.execute('grid', grid_x, grid_y)

    # 默认克里金插值
    ok_default = OrdinaryKriging(
        x, y, z,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False,
        nlags=10
    )
    default_grid, default_var = ok_default.execute('grid', grid_x, grid_y)

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)
    vmin = min(np.nanmin(pso_grid), np.nanmin(aco_grid), np.nanmin(ga_grid), np.nanmin(default_grid))
    vmax = max(np.nanmax(pso_grid), np.nanmax(aco_grid), np.nanmax(ga_grid), np.nanmax(default_grid))
    var_vmin = min(np.nanmin(pso_var), np.nanmin(aco_var), np.nanmin(ga_var), np.nanmin(default_var))
    var_vmax = max(np.nanmax(pso_var), np.nanmax(aco_var), np.nanmax(ga_var), np.nanmax(default_var))
    extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())

    # 主图 contourf
    cs0 = axes[0, 0].contourf(grid_xx, grid_yy, pso_grid, cmap='viridis', levels=50, vmin=vmin, vmax=vmax)
    axes[0, 0].scatter(x, y, c=z, edgecolor='k', s=40)
    axes[0, 0].set_title('PSO 克里金插值')


    cs1 = axes[0, 1].contourf(grid_xx, grid_yy, aco_grid, cmap='viridis', levels=50, vmin=vmin, vmax=vmax)
    axes[0, 1].scatter(x, y, c=z, edgecolor='k', s=40)
    axes[0, 1].set_title('ACO 克里金插值')


    cs2 = axes[0, 2].contourf(grid_xx, grid_yy, ga_grid, cmap='viridis', levels=50, vmin=vmin, vmax=vmax)
    axes[0, 2].scatter(x, y, c=z, edgecolor='k', s=40)
    axes[0, 2].set_title('GA 克里金插值')


    cs3 = axes[0, 3].contourf(grid_xx, grid_yy, default_grid, cmap='viridis', levels=50, vmin=vmin, vmax=vmax)
    axes[0, 3].scatter(x, y, c=z, edgecolor='k', s=40)
    axes[0, 3].set_title('默认克里金插值')


    # 方差图 imshow
    im0 = axes[1, 0].imshow(pso_var, origin='lower', extent=extent, cmap='magma', vmin=var_vmin, vmax=var_vmax, aspect='equal')
    axes[1, 0].scatter(x, y, c='white', s=10)
    axes[1, 0].set_title('PSO 方差')


    im1 = axes[1, 1].imshow(aco_var, origin='lower', extent=extent, cmap='magma', vmin=var_vmin, vmax=var_vmax, aspect='equal')
    axes[1, 1].scatter(x, y, c='white', s=10)
    axes[1, 1].set_title('ACO 方差')


    im2 = axes[1, 2].imshow(ga_var, origin='lower', extent=extent, cmap='magma', vmin=var_vmin, vmax=var_vmax, aspect='equal')
    axes[1, 2].scatter(x, y, c='white', s=10)
    axes[1, 2].set_title('GA 方差')


    im3 = axes[1, 3].imshow(default_var, origin='lower', extent=extent, cmap='magma', vmin=var_vmin, vmax=var_vmax, aspect='equal')
    axes[1, 3].scatter(x, y, c='white', s=10)
    axes[1, 3].set_title('默认方差')


    # colorbar
    cbar0 = fig.colorbar(cs1, ax=axes[0, :], orientation='horizontal', fraction=0.05, pad=0.18, aspect=40, shrink=0.95)
    cbar0.set_label('插值值')
    cbar1 = fig.colorbar(im1, ax=axes[1, :], orientation='horizontal', fraction=0.05, pad=0.18, aspect=40, shrink=0.95)
    cbar1.set_label('方差')

    plt.suptitle(f"四种克里金插值与方差对比（{target_layer}）", fontsize=18)
    plt.savefig(f"./pic/{target_layer}kriging_comparison.png", dpi=300)
    plt.show()

    # 新增：对比绘制四种全点对半方差-距离图和分组均值点拟合图
    fig2, axes2 = plt.subplots(2, 4, figsize=(28, 12), constrained_layout=True)
    # 第一行：全部点对
    plot_all_pairs_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters={'nugget': pso_params[0], 'range': pso_params[1], 'sill': pso_params[2]}, ax=axes2[0, 0], label='PSO')
    plot_all_pairs_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters={'nugget': aco_params[0], 'range': aco_params[1], 'sill': aco_params[2]}, ax=axes2[0, 1], label='ACO')
    plot_all_pairs_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters={'nugget': ga_params[0], 'range': ga_params[1], 'sill': ga_params[2]}, ax=axes2[0, 2], label='GA')
    default_nugget = ok_default.variogram_model_parameters[0]
    default_range = ok_default.variogram_model_parameters[1]
    default_sill = ok_default.variogram_model_parameters[2]
    plot_all_pairs_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters={'nugget': default_nugget, 'range': default_range, 'sill': default_sill}, ax=axes2[0, 3], label='未优化')
    axes2[0, 0].set_title('PSO 全点对')
    axes2[0, 1].set_title('ACO 全点对')
    axes2[0, 2].set_title('GA 全点对')
    axes2[0, 3].set_title('未优化全点对')
    # 第二行：分组均值点
    plot_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters={'nugget': pso_params[0], 'range': pso_params[1], 'sill': pso_params[2]}, nlags=10, ax=axes2[1, 0], label='PSO')
    plot_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters={'nugget': aco_params[0], 'range': aco_params[1], 'sill': aco_params[2]}, nlags=10, ax=axes2[1, 1], label='ACO')
    plot_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters={'nugget': ga_params[0], 'range': ga_params[1], 'sill': ga_params[2]}, nlags=10, ax=axes2[1, 2], label='GA')
    plot_semivariogram(x, y, z, variogram_model='spherical', variogram_parameters={'nugget': default_nugget, 'range': default_range, 'sill': default_sill}, nlags=10, ax=axes2[1, 3], label='未优化')
    axes2[1, 0].set_title('PSO 分组均值')
    axes2[1, 1].set_title('ACO 分组均值')
    axes2[1, 2].set_title('GA 分组均值')
    axes2[1, 3].set_title('未优化分组均值')
    plt.suptitle(f"四种克里金参数下的全点对与分组均值半方差-距离关系对比（{target_layer}）", fontsize=18)
    plt.show()


def main():
    # 数据路径和目标层
    data_path = "./data/随机分布钻孔数据.xlsx"
    target_layer = "粉土"
    # 生成数据
    iters = 500
    pso_optimizer = PSO_Krige_Optimizer(iters)
    aco_optimizer = ACO_Krige_Optimizer(iters)
    ga_optimizer = GA_Krige_Optimizer(iters)

    train_data, test_data = pso_optimizer.generate_data(data_path, seed=0, target_layer=target_layer)

    # 自动定义参数空间
    nuggets_range, ranges_range, sills_range = pso_optimizer.define_parameter_space(*train_data)

    # PSO 优化
    pso_optimizer.particle_swarm_optimize(*train_data, *test_data, nuggets_range, ranges_range, sills_range)
    print("PSO 最优参数:", list(pso_optimizer.get_parameter()))

    # ACO 优化
    aco_optimizer.ant_colony_optimize(*train_data, *test_data, nuggets_range, ranges_range, sills_range)
    print("ACO 最优参数:", list(aco_optimizer.get_parameter()))

    # GA 优化
    ga_optimizer.genetic_optimize(*train_data, *test_data, nuggets_range, ranges_range, sills_range)
    print("GA 最优参数:", list(ga_optimizer.get_parameter()))

    # # 可视化三种克里金插值及其方差
    # df = pd.read_excel(data_path)
    # layer_df = df[df["地层"] == target_layer]
    # x = layer_df["X"].values.astype(np.float64)
    # y = layer_df["Y"].values.astype(np.float64)
    # z = layer_df["厚度"].values.astype(np.float64)
    visualize_kriging_results(pso_optimizer, aco_optimizer, ga_optimizer, target_layer, train_data)

if __name__ == "__main__":
    main()