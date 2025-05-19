import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_excel("钻孔数据.xlsx")
# print(df.head())
layers = df["地层"].unique()
np.random.seed(17)
predict_potins = 100
new_points = [(df["X"].min() + np.random.rand() * (df["X"].max() - df["X"].min()),
                df["Y"].min() + np.random.rand() * (df["Y"].max() - df["Y"].min())) for _ in range(predict_potins)]


def get_newpoints_by_model(model,x,y,z,new_points):
    predicted_data = []
    OK = OrdinaryKriging(
        x, y, z,
        variogram_model=model,
        verbose=False,
        enable_plotting=False
    )
    for i, (px, py) in enumerate(new_points):
        z_pred, z_var = OK.execute('points', np.array([px], dtype=np.float64), np.array([py], dtype=np.float64))
        predicted_data.append({
            "钻孔编号": f"NewZK{i+1}",
            "X": px,
            "Y": py,
            "地层": layer,
            "厚度": float(z_pred[0])
        })
    return predicted_data


layer = "填土"
layer_df = df[df["地层"] == layer]

x = layer_df["X"].values.astype(np.float64)
y = layer_df["Y"].values.astype(np.float64)
z = layer_df["厚度"].values.astype(np.float64)

sprical_predicted_data = get_newpoints_by_model("spherical",x,y,z,new_points)
linear_predicted_data = get_newpoints_by_model("linear",x,y,z,new_points)
gaussian_predicted_data = get_newpoints_by_model("gaussian",x,y,z,new_points)
power_predicted_data = get_newpoints_by_model("power",x,y,z,new_points)

# 只标注一次预测点位置
plt.figure(figsize=(18, 12))
plt.subplot(2, 2, 1)
plt.scatter(x, y, c='black', label='原始钻孔', s=60)
plt.scatter(
    [d["X"] for d in sprical_predicted_data],
    [d["Y"] for d in sprical_predicted_data],
    c='red', marker='^', label='预测点', s=40
)
plt.xlabel("X 坐标")
plt.ylabel("Y 坐标")
plt.title("钻孔与预测点分布")
plt.legend()
plt.grid(True)

# 厚度可视化
def plot_thickness(ax, predicted_data, title, color):
    ax.scatter([d["X"] for d in predicted_data], [d["Y"] for d in predicted_data],
               c=[d["厚度"] for d in predicted_data], cmap=color, marker='s', s=60)
    ax.set_xlabel("X 坐标")
    ax.set_ylabel("Y 坐标")
    ax.set_title(title)
    plt.colorbar(ax.collections[0], ax=ax, label="厚度")
    ax.grid(True)

ax2 = plt.subplot(2, 2, 2)
plot_thickness(ax2, sprical_predicted_data, "球状(spherical)厚度分布", "Reds")

ax3 = plt.subplot(2, 2, 3)
plot_thickness(ax3, power_predicted_data, "幂(power)厚度分布", "Blues")

ax4 = plt.subplot(2, 2, 4)
plot_thickness(ax4, gaussian_predicted_data, "高斯(gaussian)厚度分布", "Greens")

plt.tight_layout()
plt.savefig("厚度分布对比.png", dpi=300)
plt.show()

# 新增：折线图展示不同模型预测点的厚度
plt.figure(figsize=(10, 6))
plt.plot(
    range(len(sprical_predicted_data)),
    [d["厚度"] for d in sprical_predicted_data],
    label="球状(spherical)", marker='o'
)
plt.plot(
    range(len(linear_predicted_data)),
    [d["厚度"] for d in linear_predicted_data],
    label="线性(linear)", marker='s'
)
plt.plot(
    range(len(gaussian_predicted_data)),
    [d["厚度"] for d in gaussian_predicted_data],
    label="高斯(gaussian)", marker='^'
)
plt.plot(
    range(len(power_predicted_data)),
    [d["厚度"] for d in power_predicted_data],
    label="幂(power)", marker='x'
)
plt.xlabel("预测点编号")
plt.ylabel("厚度")
plt.title("不同变差函数预测点厚度折线图")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("厚度折线图.png", dpi=300)
plt.show()






