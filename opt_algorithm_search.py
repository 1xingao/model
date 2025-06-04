import numpy as np
import matplotlib.pyplot as plt
from opt.pso_krige_opt import PSO_Krige_Optimizer
from opt.aco_krige_opt import ACO_Krige_Optimizer
import src.default_krige as Default_Krige
from pykrige.ok import OrdinaryKriging


def visualize_kriging_results(pso_optimizer, aco_optimizer, default_krige, train_data):
    # 获取参数
    pso_params = pso_optimizer.get_parameter()
    aco_params = aco_optimizer.get_parameter()

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

    # 默认克里金插值
    ok_default = OrdinaryKriging(
        x, y, z,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )
    default_grid, default_var = ok_default.execute('grid', grid_x, grid_y)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    vmin = min(np.nanmin(pso_grid), np.nanmin(aco_grid), np.nanmin(default_grid))
    vmax = max(np.nanmax(pso_grid), np.nanmax(aco_grid), np.nanmax(default_grid))
    var_vmin = min(np.nanmin(pso_var), np.nanmin(aco_var), np.nanmin(default_var))
    var_vmax = max(np.nanmax(pso_var), np.nanmax(aco_var), np.nanmax(default_var))
    extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())

    # 主图 contourf
    cs0 = axes[0, 0].contourf(grid_xx, grid_yy, pso_grid, cmap='viridis', levels=50, vmin=vmin, vmax=vmax)
    axes[0, 0].scatter(x, y, c=z, edgecolor='k', s=40)
    axes[0, 0].set_title('PSO 克里金插值')


    cs1 = axes[0, 1].contourf(grid_xx, grid_yy, aco_grid, cmap='viridis', levels=50, vmin=vmin, vmax=vmax)
    axes[0, 1].scatter(x, y, c=z, edgecolor='k', s=40)
    axes[0, 1].set_title('ACO 克里金插值')


    cs2 = axes[0, 2].contourf(grid_xx, grid_yy, default_grid, cmap='viridis', levels=50, vmin=vmin, vmax=vmax)
    axes[0, 2].scatter(x, y, c=z, edgecolor='k', s=40)
    axes[0, 2].set_title('默认克里金插值')


    # 方差图 imshow
    im0 = axes[1, 0].imshow(pso_var, origin='lower', extent=extent, cmap='magma', vmin=var_vmin, vmax=var_vmax, aspect='equal')
    axes[1, 0].scatter(x, y, c='white', s=10)
    axes[1, 0].set_title('PSO 方差')


    im1 = axes[1, 1].imshow(aco_var, origin='lower', extent=extent, cmap='magma', vmin=var_vmin, vmax=var_vmax, aspect='equal')
    axes[1, 1].scatter(x, y, c='white', s=10)
    axes[1, 1].set_title('ACO 方差')


    im2 = axes[1, 2].imshow(default_var, origin='lower', extent=extent, cmap='magma', vmin=var_vmin, vmax=var_vmax, aspect='equal')
    axes[1, 2].scatter(x, y, c='white', s=10)
    axes[1, 2].set_title('默认方差')


    # colorbar
    cbar0 = fig.colorbar(cs0, ax=axes[0, :], orientation='horizontal', fraction=0.05, pad=0.18, aspect=40, shrink=0.95)
    cbar0.set_label('插值值')
    cbar1 = fig.colorbar(im0, ax=axes[1, :], orientation='horizontal', fraction=0.05, pad=0.18, aspect=40, shrink=0.95)
    cbar1.set_label('方差')

    plt.suptitle("三种克里金插值与方差对比", fontsize=18)

    plt.show()


def main():
    # 设置随机种子以确保结果可重复
    np.random.seed(0)

    # 数据路径和目标层
    data_path = "./data/增强后的钻孔数据.xlsx"
    target_layer = "砾石"

    # 生成数据
    pso_optimizer = PSO_Krige_Optimizer()
    aco_optimizer = ACO_Krige_Optimizer()
    default_krige = Default_Krige.Default_Krige()

    train_data, test_data = pso_optimizer.generate_data(data_path, target_layer=target_layer)
    
    # 自动定义参数空间
    nuggets_range, ranges_range, sills_range = pso_optimizer.define_parameter_space(*train_data)

    # PSO 优化
    pso_optimizer.particle_swarm_optimize(*train_data, *test_data, nuggets_range, ranges_range, sills_range)
    print("PSO 最优参数:", list(pso_optimizer.get_parameter()))

    # ACO 优化
    aco_optimizer.ant_colony_optimize(*train_data, *test_data, nuggets_range, ranges_range, sills_range)
    print("ACO 最优参数:", list(aco_optimizer.get_parameter()))

    # # 原始参数
    # default_krige.fit_default(*train_data)
    # default_params = default_krige.get_parameter()
    # print("默认参数:", default_params)

    # 可视化三种克里金插值及其方差
    visualize_kriging_results(pso_optimizer, aco_optimizer, default_krige, train_data)

if __name__ == "__main__":
    main()