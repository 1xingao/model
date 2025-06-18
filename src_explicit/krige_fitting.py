import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

# 已知数据点的坐标和对应值
np.random.seed(17)
n_points = 30
pot = np.random.uniform(0, 10, (n_points, 2))  # 随机 2D 空间坐标
z = np.sin(pot[:, 0] / 2) + np.cos(pot[:, 1] / 3) + np.random.normal(0, 0.2, n_points)  # 模拟属性值


OK = OrdinaryKriging(
    pot[:,0],pot[:,1] ,z,
    variogram_model="spherical",  
    enable_plotting=True
)

# 对目标点 P(0.25, 0.25) 进行插值
z_interp, ss = OK.execute("points", [0.25], [0.25])

print(f"插值结果 z(0.25, 0.25) = {z_interp[0]:.2f}")
print(f"插值估计误差方差 = {ss[0]:.4f}")
