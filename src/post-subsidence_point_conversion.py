import numpy as np
import pandas as pd
from scipy.special import erf
import os
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation    
from matplotlib.patches import Patch
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
TEST_CSV_PATH = os.path.join("data", "test_layers_points.csv")

def generate_test_layers_csv(num_points: int = 1000, path: str = TEST_CSV_PATH, random_state: int = 42):
    """
    生成用于多层沉陷计算的测试点数据并保存为 CSV。
    格式: Layer, X, Y, Z （共 num_points 行, 多个层共计）

    设计原则:
    - 三层: Bedrock < Coal < Topsoil
    - 在同一 (X,Y) 上各层 Z 递增
    - Z 使用平滑基面 + 小扰动, 保持层间平均厚度稳定
    - 总点数约为 num_points (在三层之间平均分配)
    """
    rng = np.random.default_rng(random_state)
    layers = ["Bedrock", "Coal", "Topsoil"]
    n_each = [num_points // 3] * 3
    n_each[-1] += num_points - sum(n_each)  # 分配余数

    records = []
    # 定义基础面的函数（与示例相似, 但参数稍作变化）
    def base_surface(x, y):
        return 2.0 * np.sin(2 * np.pi * x / 600.0) + 1.5 * np.cos(2 * np.pi * y / 400.0)

    # 平均层间距（厚度）
    coal_offset = 30.0  # 相对基岩平均高程
    topsoil_offset = 60.0  # 相对基岩平均高程

    # 为避免同一 (X,Y) 三层重复排布(使点更均匀)，对每层独立采样
    for lname, n in zip(layers, n_each):
        X = rng.uniform(0, 500, size=n)
        Y = rng.uniform(0, 300, size=n)
        base = base_surface(X, Y)
        noise = rng.normal(0, 0.8, size=n)
        if lname == "Bedrock":
            Z = base + noise
        elif lname == "Coal":
            Z = base + coal_offset + noise
        else:  # Topsoil
            Z = base + topsoil_offset + noise
        for x, y, z in zip(X, Y, Z):
            records.append((lname, x, y, z))

    df = pd.DataFrame(records, columns=["Layer", "X", "Y", "Z"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8')
    return path

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


def plot_surfaces(layers_original: dict, layers_after: dict, value_after_col: str = 'Z_subsided', out_before='surfaces_before.png', out_after='surfaces_after.png'):
    order_layers = sorted(layers_original.keys(), key=lambda n: layers_original[n]['Z'].mean())
    colors = ['#2ca02c', '#1f77b4', '#8c564b', '#9467bd', '#d62728']

    # Before
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title('原始层面 (Before)')
    for i, lname in enumerate(order_layers):
        df_layer = layers_original[lname]
        if len(df_layer) < 3:
            continue
        tri = Triangulation(df_layer['X'], df_layer['Y'])
        ax1.plot_trisurf(tri, df_layer['Z'], color=colors[i % len(colors)], alpha=0.7, linewidth=0.1, edgecolor='none', label=lname)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=30, azim=-60)
    # 手动图例（plot_trisurf 不自动进入 legend）

    patches = [Patch(facecolor=colors[i % len(colors)], label=l) for i, l in enumerate(order_layers)]
    ax1.legend(handles=patches, loc='best')
    plt.tight_layout()

    plt.show()


    # After
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_title('下沉后层面 (After)')
    for i, lname in enumerate(order_layers):
        df_layer = layers_after[lname]
        if len(df_layer) < 3:
            continue
        tri = Triangulation(df_layer['X'], df_layer['Y'])
        ax2.plot_trisurf(tri, df_layer[value_after_col], color=colors[i % len(colors)], alpha=0.7, linewidth=0.1, edgecolor='none', label=lname)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=30, azim=-60)
    patches = [Patch(facecolor=colors[i % len(colors)], label=l) for i, l in enumerate(order_layers)]
    ax2.legend(handles=patches, loc='best')
    plt.tight_layout()

    plt.show()

# ==== 示例 ====
if __name__ == "__main__":
    # 1. 若不存在测试数据 CSV，则生成 1000 点测试数据
    if not os.path.exists(TEST_CSV_PATH):
        generate_test_layers_csv(num_points=1000, path=TEST_CSV_PATH)
        print(f"已生成测试数据: {TEST_CSV_PATH}")
    else:
        print(f"发现已有测试数据: {TEST_CSV_PATH}")

    # 2. 读取测试数据并按 Layer 切分
    df_all = pd.read_csv(TEST_CSV_PATH)
    if not set(["Layer","X","Y","Z"]).issubset(df_all.columns):
        raise ValueError("测试 CSV 缺少必须列: Layer,X,Y,Z")

    layers_data = {lname: sub_df[["X","Y","Z"]].reset_index(drop=True)
                   for lname, sub_df in df_all.groupby("Layer")}

    # 若缺层处理
    expected_layers = ["Bedrock","Coal","Topsoil"]
    for el in expected_layers:
        if el not in layers_data:
            raise ValueError(f"缺少层 {el} 的数据，无法继续")

    # 3. 定义层类型 (Topsoil 视为松散层)
    layer_types = {lname: ("loose" if lname.lower()=="topsoil" else "rock")
                   for lname in layers_data.keys()}

    # 4. 选定采矿层 (这里示例选择 Coal)
    mining_layer_name = "Coal"
    if mining_layer_name not in layers_data:
        raise ValueError("测试数据中不含采矿层 'Coal'")
    mining_layer_depth = layers_data[mining_layer_name]['Z'].mean()

    # 5. 基础参数（可按需要修改）
    q_base = 0.9
    H_base = 150.0
    tan_beta_base = np.tan(np.deg2rad(45))
    theta0_base = np.deg2rad(70)
    # 取工作面中心为数据 X,Y 的均值
    cx = df_all['X'].mean()
    cy = df_all['Y'].mean()
    # 估计工作面长宽为数据范围 * 某比例
    Lx = (df_all['X'].max() - df_all['X'].min()) * 0.2
    Ly = (df_all['Y'].max() - df_all['Y'].min()) * 0.2
    alpha = np.deg2rad(0)  # 不旋转

    # 6. 执行多层沉陷计算
    results = subsidence_multilayer(
        layers_data, mining_layer_name, mining_layer_depth,
        q_base, H_base, tan_beta_base, theta0_base,
        cx, cy, Lx, Ly, alpha,
        layer_types
    )

    # 7. 合并输出为单一文件（格式同输入: Layer,X,Y,Z；其中 Z 为下沉后高程）
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    merged_records = []
    for layer, df in results.items():
        tmp = df[['X','Y','Z_subsided']].copy()
        tmp.rename(columns={'Z_subsided':'Z'}, inplace=True)
        tmp.insert(0, 'Layer', layer)
        merged_records.append(tmp)
        print(f"{layer} 计算完成：点数 {len(df)} 平均沉陷 {df['Subsidence'].mean():.3f} m")
    merged_df = pd.concat(merged_records, ignore_index=True)
    merged_out_path = os.path.join(out_dir, 'layers_subsided_merged.csv')
    merged_df.to_csv(merged_out_path, index=False, encoding='utf-8')
    print(f"已写入合并结果文件: {merged_out_path} (列: Layer,X,Y,Z)")

    # 8. 计算前后可视化（matplotlib 三角网表面）
    try:
        plot_surfaces(layers_data, results)
    except Exception as e:
        print(f"[警告] matplotlib 可视化失败: {e}")
