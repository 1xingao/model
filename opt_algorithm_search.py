import numpy as np
import matplotlib.pyplot as plt
from opt.pso_krige_opt import PSO_Krige_Optimizer
from opt.aco_krige_opt import ACO_Krige_Optimizer
from opt.ga_krige_opt import GA_Krige_Optimizer
import src.default_krige as Default_Krige
from pykrige.ok import OrdinaryKriging
import pandas as pd


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
        enable_plotting=False
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

    plt.show()


def main():
    # 数据路径和目标层
    data_path = "./data/随机分布钻孔数据.xlsx"
    target_layer = "黄土"
    # 生成数据
    pso_optimizer = PSO_Krige_Optimizer()
    aco_optimizer = ACO_Krige_Optimizer()
    ga_optimizer = GA_Krige_Optimizer()

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