import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

plt.rcParams['font.sans-serif'] = ['SimHei']

def model_krige(model,df):

    # print(df.head())
    layers = df["地层"].unique()
    np.random.seed(17)
    predict_potins = 50
    new_points = [(df["X"].min() + np.random.rand() * (df["X"].max() - df["X"].min()),
                    df["Y"].min() + np.random.rand() * (df["Y"].max() - df["Y"].min())) for _ in range(predict_potins)]

    predicted_data = []


    for layer in layers:
        layer_df = df[df["地层"] == layer]
        x = layer_df["X"].values.astype(np.float64)
        y = layer_df["Y"].values.astype(np.float64)
        z = layer_df["厚度"].values.astype(np.float64)

        OK = OrdinaryKriging(
            x, y, z,
            variogram_model=model,
            verbose=True,
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
    # 创建 DataFrame 并与原始数据合并
def visualization_data(predicted_data,df):
    predicted_df = pd.DataFrame(predicted_data)
    combined_df = pd.concat([df, predicted_df], ignore_index=True)

    # 按钻孔编号分组展示
    grouped_predicted_df = predicted_df.groupby("钻孔编号")

    plt.figure(figsize=(20, 8))
    plt.subplot(1,2,1)
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


    plt.subplot(1,2,2)

    color_map = {
        "填土" :"red",
        "黄土": "yellow",
        "粉土": "green",
        "泥岩": "brown",
        "砾石": "purple",
        "砂岩": "pink",
        "粉砂": "lightblue",
    }
    for target_layer in df["地层"].unique():

        layer_df = combined_df[combined_df["地层"] == target_layer]

        z = layer_df["厚度"].values.astype(np.float64)
        
        plt.plot(layer_df["钻孔编号"], z, label=target_layer, color=color_map[target_layer], marker="o", markersize=5, linestyle="-")


    plt.xlabel("钻孔编号")
    plt.ylabel("厚度 (m)")
    plt.title(f"厚度分布")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    image_path = "./pic/钻孔分布图.png"
    excel_path = "./pic/预测钻孔数据.xlsx"
    # plt.savefig(image_path)
    plt.show()
    # predicted_df.to_excel(excel_path, index=False)


if __name__ == "__main__":
    df = pd.read_excel("./data/钻孔数据.xlsx")
    predicted_data = model_krige("spherical",df)
    visualization_data(predicted_data,df)
    