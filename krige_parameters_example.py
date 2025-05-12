import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit

# 1. 模拟一组空间点与属性值（模拟实际测量数据）
np.random.seed(17)
n_points = 30
x = np.random.uniform(0, 10, (n_points, 2))  # 随机 2D 空间坐标
z = np.sin(x[:, 0] / 2) + np.cos(x[:, 1] / 3) + np.random.normal(0, 0.2, n_points)  # 模拟属性值

# 2. 计算实验变异函数
# 2.1 计算距离矩阵和差值平方矩阵
dists = squareform(pdist(x))
diffs = squareform(pdist(z.reshape(-1, 1), metric='sqeuclidean'))

# 2.2 分桶求平均 -> 实验变异函数（半方差）
bins = np.linspace(0, np.max(dists), 15)
bin_centers = (bins[:-1] + bins[1:]) / 2
gamma_exp = []

for i in range(len(bins) - 1):
    mask = (dists >= bins[i]) & (dists < bins[i + 1])
    if np.any(mask):
        gamma_exp.append(np.mean(diffs[mask]) / 2)
    else:
        gamma_exp.append(np.nan)

gamma_exp = np.array(gamma_exp)

# 3. 定义球状模型
def spherical_model(h, nugget, sill, range_):
    h = np.array(h)
    gamma = np.where(
        h <= range_,
        nugget + sill * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
        nugget + sill
    )
    return gamma

# 4. 拟合模型
valid = ~np.isnan(gamma_exp)
popt, _ = curve_fit(spherical_model, bin_centers[valid], gamma_exp[valid],
                    bounds=([0, 0, 0.1], [1, 5, 20]))  # 约束范围

# 5. 画图展示
h_fine = np.linspace(0, np.max(dists), 200)
gamma_fit = spherical_model(h_fine, *popt)

plt.figure(figsize=(8, 5))
plt.scatter(bin_centers, gamma_exp, color='blue', label='实验变异函数')
plt.plot(h_fine, gamma_fit, color='red', label='拟合球状模型')
plt.xlabel('距离 h')
plt.ylabel('γ(h)')
plt.title('实验变异函数及其球状模型拟合')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(), popt
