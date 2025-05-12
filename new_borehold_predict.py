import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_excel("钻孔数据.xlsx")
# print(df.head())
layers = df["地层"].unique()

new_points = [(150, 220), (190, 210), (230, 230), (270, 210)]

predicted_data = []

for i, (px, py) in enumerate(new_points):
    for layer in layers:
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

        z_pred, z_var = OK.execute('points', np.array([px], dtype=np.float64), np.array([py], dtype=np.float64))
        predicted_data.append({
            "钻孔编号": f"NewZK{i+1}",
            "X": px,
            "Y": py,
            "地层": layer,
            "厚度": float(z_pred[0])
        })

# 创建 DataFrame 并与原始数据合并
predicted_df = pd.DataFrame(predicted_data)
combined_df = pd.concat([df, predicted_df], ignore_index=True)

# 按钻孔编号分组展示
grouped_predicted_df = predicted_df.groupby("钻孔编号")

plt.figure(figsize=(10, 8))
for zk in df["钻孔编号"].unique():
    zk_df = df[df["钻孔编号"] == zk]
    plt.scatter(zk_df["X"].iloc[0], zk_df["Y"].iloc[0], color="blue", label="原始钻孔" if zk == "ZK01" else "", s=60)

for zk, group in grouped_predicted_df:
    plt.scatter(group["X"].iloc[0], group["Y"].iloc[0], color="red", marker="^", label="预测钻孔" if zk == "NewZK1" else "", s=80)

plt.xlabel("X 坐标")
plt.ylabel("Y 坐标")
plt.title("原始与预测钻孔位置分布")
plt.legend()
plt.grid(True)
plt.tight_layout()

image_path = "钻孔分布图.png"
excel_path = "预测钻孔数据.xlsx"
plt.savefig(image_path)
predicted_df.to_excel(excel_path, index=False)

image_path, excel_path
