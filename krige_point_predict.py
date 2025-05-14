import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging

# Step 1: 读取 Excel 文件
df = pd.read_excel("钻孔数据.xlsx")

# Step 2: 提取目标地层（例如 "黄土"）的坐标和厚度
target_layer = "黄土"
layer_df = df[df["地层"] == target_layer]

# 获取坐标和厚度
x = layer_df["X"].values.astype(np.float64)
y = layer_df["Y"].values.astype(np.float64)
z = layer_df["厚度"].values.astype(np.float64)

# Step 3: 创建普通克里金对象
OK = OrdinaryKriging(
    x, y, z,
    variogram_model='spherical',
    verbose=False,
    enable_plotting=False
)

# Step 4: 定义新钻孔位置（转为 float）
new_x = np.array([150], dtype=np.float64)
new_y = np.array([220], dtype=np.float64)

# 执行插值
z_pred, z_var = OK.execute('points', new_x, new_y)

print(f"在位置 ({new_x[0]}, {new_y[0]}) 处预测地层“{target_layer}”厚度为: {z_pred[0]:.2f} 米")
