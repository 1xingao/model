import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
import build_block_pyvista

upper_tin = pd.read_csv('../data/upper_tin.csv')  # 上层点集
lower_tin = pd.read_csv('../data/lower_tin.csv')  # 下层点集


x_min = min(upper_tin['x'].min(), lower_tin['x'].min())
x_max = max(upper_tin['x'].max(), lower_tin['x'].max())
y_min = min(upper_tin['y'].min(), lower_tin['y'].min())
y_max = max(upper_tin['y'].max(), lower_tin['y'].max())

n_x, n_y = 50, 50 
xi = np.linspace(x_min, x_max, n_x)
yi = np.linspace(y_min, y_max, n_y)
grid_x, grid_y = np.meshgrid(xi, yi)
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]


ok_upper = OrdinaryKriging(
    upper_tin['x'], upper_tin['y'], upper_tin['z'], variogram_model='linear', verbose=False, enable_plotting=False)
z_upper, _ = ok_upper.execute('points', grid_points[:, 0], grid_points[:, 1])

ok_lower = OrdinaryKriging(
    lower_tin['x'], lower_tin['y'], lower_tin['z'], variogram_model='linear', verbose=False, enable_plotting=False)
z_lower, _ = ok_lower.execute('points', grid_points[:, 0], grid_points[:, 1])

# 使用其他文件的pyvista成品进行替代
Block = build_block_pyvista.Block(np.column_stack((xi, yi)), [z_upper, z_lower])
Block.execute()
# tri = Delaunay(grid_points)
# faces = np.hstack((np.full((tri.simplices.shape[0], 1), 3), tri.simplices)).astype(np.int32)


# surf_upper = pv.PolyData(np.c_[grid_points, z_upper], faces)
# surf_lower = pv.PolyData(np.c_[grid_points, z_lower], faces)


# prisms = []
# for simplex in tri.simplices:

#     pts_upper = np.c_[grid_points[simplex], z_upper[simplex]]
#     pts_lower = np.c_[grid_points[simplex], z_lower[simplex]]
   
#     prism = pv.PolyData()
#     prism.points = np.vstack([pts_upper, pts_lower])
#     prism.faces = [3, 0, 1, 2, 3, 3, 4, 5, 3, 0, 1, 4, 5, 2, 3]  # 仅示意，具体面需按 pyvista 规范
#     prisms.append(prism)


# plotter = pv.Plotter()
# plotter.add_mesh(surf_upper, color='orange', opacity=0.7, label='上层TIN')
# plotter.add_mesh(surf_lower, color='blue', opacity=0.7, label='下层TIN')

# plotter.add_mesh(prisms, color='green', opacity=0.3)
# plotter.add_legend()
# plotter.show()

# 你可以根据实际数据格式调整数据读取和可视化部分。
# 依赖库：pykrige, scipy, pyvista, pandas, numpy
