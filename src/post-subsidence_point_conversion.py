import numpy as np
import pandas as pd
from scipy.special import erf

def subsidence_probability_integral(points_df, q, H, tan_beta, theta0, center_x, center_y, Lx, Ly, alpha=0.0):
    """
    三维概率积分法计算任意层面的下沉量
    """
    B = H * tan_beta  # 主要影响半径
    delta = H / np.tan(theta0)  # 传播角修正距离（cot(theta0)）

    dx = points_df['X'] - center_x
    dy = points_df['Y'] - center_y

    # 倾角修正
    dx_prime = dx * np.cos(alpha) + dy * np.sin(alpha)
    dy_prime = -dx * np.sin(alpha) + dy * np.cos(alpha)

    # 加入 θ0 修正的积分限
    fx = 0.5 * (
        erf((dx_prime + Lx / 2 + delta) / (np.sqrt(2) * B)) -
        erf((dx_prime - Lx / 2 - delta) / (np.sqrt(2) * B))
    )
    fy = 0.5 * (
        erf((dy_prime + Ly / 2 + delta) / (np.sqrt(2) * B)) -
        erf((dy_prime - Ly / 2 - delta) / (np.sqrt(2) * B))
    )

    subsidence = q * H * fx * fy
    points_df['Z_subsided'] = points_df['Z'] - subsidence
    points_df['Subsidence'] = subsidence
    return points_df

def parameter_transfer(H_base, tan_beta_base, theta0_base, q_base, delta_h, layer_type="rock"):
    """
    岩层沉陷参数沿岩层传递公式（刘吉波 2014 版）
    H_base: 采层深度
    tan_beta_base: 采层主要影响角正切
    theta0_base: 采层传播角
    q_base: 采层下沉系数
    delta_h: 目标层与采层的高程差（m）
    layer_type: 'rock' 或 'loose'，松散层修正 q
    """
    # 目标层采深
    H_target = H_base - delta_h
    if H_target <= 0:
        H_target = 1e-3  # 避免为零或负值

    # 主要影响半径 B 传递
    B_base = H_base * tan_beta_base
    B_target = B_base - delta_h * tan_beta_base

    # 主要影响角正切
    tan_beta_target = B_target / H_target

    # 传播角传递
    theta0_target = np.arctan(np.tan(theta0_base) * (H_base / H_target))

    # 下沉系数修正
    if layer_type == "loose":  # 松散层
        k_q = 0.9
    else:
        k_q = 1.0
    q_target = q_base * k_q

    return H_target, tan_beta_target, theta0_target, q_target

def subsidence_multilayer(layers_dict, mining_layer_name, mining_layer_depth,
                          q_base, H_base, tan_beta_base, theta0_base,
                          center_x, center_y, Lx, Ly, alpha=0.0,
                          layer_types=None):
    """
    多层沉陷计算
    layers_dict: {layer_name: DataFrame(X,Y,Z)}
    layer_types: {layer_name: 'rock' or 'loose'}  # 用于 q 修正
    """
    result_layers = {}
    for layer_name, df_points in layers_dict.items():
        layer_mean_z = df_points['Z'].mean()
        delta_h = layer_mean_z - mining_layer_depth

        layer_type = "rock"
        if layer_types and layer_name in layer_types:
            layer_type = layer_types[layer_name]

        # 沿岩层传递
        H_target, tan_beta_target, theta0_target, q_target = parameter_transfer(
            H_base, tan_beta_base, theta0_base, q_base, delta_h, layer_type
        )

        df_subsided = subsidence_probability_integral(
            df_points.copy(), q_target, H_target, tan_beta_target,
            theta0_target, center_x, center_y, Lx, Ly, alpha
        )
        result_layers[layer_name] = df_subsided

    return result_layers

# ==== 示例 ====
if __name__ == "__main__":
    # 模拟多层点集
    layers_data = {
        "Topsoil": pd.DataFrame({"X": [100,110,120], "Y": [200,210,220], "Z": [60, 61, 62]}),
        "Coal": pd.DataFrame({"X": [100,110,120], "Y": [200,210,220], "Z": [30, 31, 32]}),
        "Bedrock": pd.DataFrame({"X": [100,110,120], "Y": [200,210,220], "Z": [0, 1, 2]})
    }
    layer_types = {
        "Topsoil": "loose",
        "Coal": "rock",
        "Bedrock": "rock"
    }

    mining_layer_name = "Coal"
    mining_layer_depth = layers_data[mining_layer_name]['Z'].mean()

    q_base = 0.9
    H_base = 300.0
    tan_beta_base = np.tan(np.deg2rad(60))
    theta0_base = np.deg2rad(60)
    cx, cy = 110, 210
    Lx, Ly = 200.0, 100.0
    alpha = np.deg2rad(0)

    results = subsidence_multilayer(
        layers_data, mining_layer_name, mining_layer_depth,
        q_base, H_base, tan_beta_base, theta0_base,
        cx, cy, Lx, Ly, alpha,
        layer_types
    )

    for layer, df in results.items():
        df.to_csv(f"./data/{layer}_subsided.csv", index=False)
        print(f"{layer} 完成，平均沉陷: {df['Subsidence'].mean():.3f} m")
