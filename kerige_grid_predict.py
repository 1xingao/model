import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_excel("钻孔数据.xlsx")
layers = df["地层"].unique()

# 设置网格范围
grid_x = np.linspace(df["X"].min(), df["X"].max(), 100)
grid_y = np.linspace(df["Y"].min(), df["Y"].max(), 100)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

# 每行最多显示的子图数量
cols = 3
rows = int(np.ceil(len(layers) / cols))

fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
axs = axs.flatten()  # 统一处理为一维数组方便索引

for idx, layer in enumerate(layers):
    layer_df = df[df["地层"] == layer]
    x = layer_df["X"].values.astype(np.float64)
    y = layer_df["Y"].values.astype(np.float64)
    z = layer_df["厚度"].values.astype(np.float64)

    OK = OrdinaryKriging(
        x, y, z,
        variogram_model='spherical',
        verbose=False,
        enable_plotting=False
    )

    z_grid, ss = OK.execute('grid', grid_x, grid_y)

    ax = axs[idx]
    c = ax.contourf(grid_xx, grid_yy, z_grid, cmap='viridis')
    ax.scatter(x, y, c='red', marker='o', label='原始数据点')
    ax.set_title(f"{layer} 厚度预测图")
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    fig.colorbar(c, ax=ax, label="厚度 (m)")
    ax.grid(True)

# 如果子图比地层多，关闭多余的子图
for j in range(len(layers), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)

plt.savefig("地层厚度预测图.png", dpi=300)
plt.show()

