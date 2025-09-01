import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from build_block_pyvista import Block
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
DATA_PATH = './data/real_data/地层坐标.xlsx'  # 输入数据: layer,x,y,z 
LOCATION_PATH = "./data/real_data/钻孔位置统计.xlsx" 
GRID_NX = 80
GRID_NY = 80
DEFAULT_VARIOGRAM = 'spherical'            # 默认变差函数模型
LAYER_VARIOGRAM = {}                    # 可为特定层指定: {'Topsoil':'spherical','Coal':'linear'}
VERBOSE_KRIGE = False
layer_name__list = ["sandstone_3","coal5-3","sandstone_2","coal5-2","sandstone_1",
                    "coal4-2","Sandstone and mudstone mixed layer","coal3-1",
                    "gravel sandstone layer","loose layer"]
Z_SCALE = 10



def load_layer_points(path: str):
    """
    读取地层坐标数据，文件格式为地层名称、x、y、z。
    参数:
        path: 文件路径
    返回:
        字典，key是地层名称，value是包含 x, y, z 的 DataFrame。
    """
    # 读取 Excel 文件
    df = pd.read_excel(path)

    # 检查必要列是否存在
    required_columns = {'地层名称', 'x', 'y', 'z'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"文件缺少必要列: {required_columns}")

    # 重命名列以统一处理
    df = df.rename(columns={'地层名称': 'layer'})

    # 删除缺失值
    df = df.dropna(subset=['layer', 'x', 'y', 'z'])

    # 按地层名称分组
    groups = {layer: group[['x', 'y', 'z']].reset_index(drop=True) for layer, group in df.groupby('layer')}
    return groups

def load_borehole_locations(path: str):
    """
    读取钻孔位置和名称。
    参数:
        path: 文件路径
    返回:
        DataFrame，包含 x, y 坐标和钻孔名称。
    """
    df = pd.read_excel(path)

    # 检查必要列是否存在
    required_columns = {'钻孔名称', 'x', 'y'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"文件缺少必要列: {required_columns}")

    # 删除缺失值
    df = df.dropna(subset=['钻孔名称', 'x', 'y'])

    return df

def build_unified_grid(layers: dict):
    xs = []
    ys = []
    for df in layers.values():
        xs.append(df['x'])
        ys.append(df['y'])
    x_min = min(s.min() for s in xs)
    x_max = max(s.max() for s in xs)
    y_min = min(s.min() for s in ys)
    y_max = max(s.max() for s in ys)
    print(f"生成模型的面积：{((x_max - x_min) * (y_max - y_min))/1000000}km²,长度：{(y_max - y_min)/1000}km,宽度：{(x_max - x_min)/1000}km  ")
    xi = np.linspace(x_min, x_max, GRID_NX)
    yi = np.linspace(y_min, y_max, GRID_NY)
    gx, gy = np.meshgrid(xi, yi)
    grid_points = np.c_[gx.ravel(), gy.ravel()]
    return xi, yi, grid_points



def build_random_grid(layers: dict, num_points: int):
    """
    生成范围内随机分布的点。
    参数:
        layers: 包含 x, y 坐标的地层数据。
        num_points: 随机生成的点数量。
    返回:
        随机分布的点坐标。
    """
    xs = []
    ys = []
    for df in layers.values():
        xs.append(df['x'])
        ys.append(df['y'])
    x_min = min(s.min() for s in xs)
    x_max = max(s.max() for s in xs)
    y_min = min(s.min() for s in ys)
    y_max = max(s.max() for s in ys)

    print(f"生成随机模型的范围：面积：{((x_max - x_min) * (y_max - y_min))/1000000}km², 长度：{(y_max - y_min)/1000}km, 宽度：{(x_max - x_min)/1000}km")

    random_x = np.random.uniform(x_min, x_max, num_points)
    random_y = np.random.uniform(y_min, y_max, num_points)
    random_points = np.c_[random_x, random_y]

    return random_points



def krige_layer(df_layer: pd.DataFrame, grid_points: np.ndarray, variogram_model: str,opt_params: list=[]):
    if len(df_layer) < 3:
        raise ValueError('点数不足，无法克里金插值 (>=3)')
    
    if len(opt_params) != 0:
        ok = OrdinaryKriging(
            df_layer['x'], df_layer['y'], df_layer['z']*10,
            variogram_model=variogram_model,
            variogram_parameters={'nugget': opt_params[0], 'range': opt_params[1], 'sill': opt_params[2]},
            verbose=VERBOSE_KRIGE,
        enable_plotting=False
        )
    else:
        ok = OrdinaryKriging(
            df_layer['x'], df_layer['y'], df_layer['z'],
            variogram_model=variogram_model,
            verbose=VERBOSE_KRIGE,
            enable_plotting=False
            )
    z_pred, _ = ok.execute('points', grid_points[:,0], grid_points[:,1])
    
    return np.asarray(z_pred)



def interpolate_all_layers(layer_points: dict, grid_points: np.ndarray):
    # 层按平均 Z 升序排列 (自下而上建模)
    order = sorted(layer_points.keys(), key=lambda k: layer_points[k]['z'].mean())
    z_list = []
    for lname in order:
        model = LAYER_VARIOGRAM.get(lname, DEFAULT_VARIOGRAM)
        z_vals = krige_layer(layer_points[lname], grid_points, model)*Z_SCALE
        z_list.append(z_vals)
    return order, z_list



def build_block_model(grid_points: np.ndarray, z_list: list, layer_names: list):
    """
    构建块体模型，并将地层名称写入模型。
    参数:
        grid_points: 网格点坐标。
        z_list: 每个地层的 z 值列表。
        layer_names: 地层名称列表。
    """
    # 创建块体模型
    block = Block(xy=grid_points, z_list=z_list)

    # 将地层名称写入模型
    block.layer_names = layer_name__list

    # 执行模型构建
    block.execute()
    block.export_model("./data/output_model.vtm")



def plot_kriging_results(z_list, xi, yi, layer_names, layer_points, borehole_locations):
    """
    显示所有克里金插值结果图在一张大图中，并标注钻孔位置和名称。
    参数:
        z_list: 每个地层的插值结果。
        xi: 网格的 x 坐标。
        yi: 网格的 y 坐标。
        layer_names: 地层名称列表。
        layer_points: 原始钻孔数据，字典形式 {layer_name: DataFrame(x, y, z)}。
        borehole_locations: DataFrame，包含钻孔位置和名称。
    """
    num_layers = len(z_list)
    cols = 3  # 每行显示的图像数量
    rows = (num_layers + cols - 1) // cols  # 计算行数

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i, z_vals in enumerate(z_list):
        ax = axes[i]
        zi = z_vals.reshape(len(yi), len(xi))
        c = ax.contourf(xi, yi, zi, cmap='viridis')
        fig.colorbar(c, ax=ax)
        ax.set_title(f"{layer_names[i]} 插值结果")
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")

        # 添加钻孔位置和名称
        ax.scatter(borehole_locations['x'], borehole_locations['y'], color='red', label='钻孔位置')
        for _, row in borehole_locations.iterrows():
            ax.text(row['x'], row['y'], row['钻孔名称'], fontsize=8, color='black')

        ax.legend()

    # 隐藏多余的子图
    for j in range(num_layers, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def validate_interpolation(layer_points, grid_points, z_list, layer_names):
    """
    验证插值结果与原始钻孔数据的误差。
    参数:
        layer_points: 原始钻孔数据，字典形式 {layer_name: DataFrame(x, y, z)}。
        grid_points: 插值网格点坐标。
        z_list: 插值结果，每层的 z 值列表。
        layer_names: 地层名称列表。
    """
    errors = []

    for i, layer_name in enumerate(layer_names):
        if layer_name not in layer_points:
            continue

        # 原始钻孔数据
        df = layer_points[layer_name]
        original_points = df[['x', 'y']].values
        original_z = df['z'].values

        # 插值结果
        interpolated_z = griddata(grid_points, z_list[i], original_points, method='linear')/10

        # 计算误差
        error = interpolated_z - original_z
        errors.append((layer_name, error))

        # 可视化对比
        plt.figure(figsize=(10, 6))
        plt.scatter(original_z, interpolated_z, alpha=0.7, label='插值 vs 原始')
        plt.plot([original_z.min(), original_z.max()], [original_z.min(), original_z.max()], 'r--', label='理想线')
        plt.xlabel('原始 Z 值')
        plt.ylabel('插值 Z 值')
        plt.title(f'{layer_name} 插值验证')
        plt.legend()
        plt.grid()
        plt.show()

    # 输出误差统计
    for layer_name, error in errors:
        print(f'层 {layer_name} 的误差统计:')
        print(f'  平均误差: {error.mean():.4f}')
        print(f'  均方误差: {(error**2).mean():.4f}')
        print(f'  最大误差: {error.max():.4f}')
        print(f'  最小误差: {error.min():.4f}')

def main():
    print(f'读取数据: {DATA_PATH}')
    layer_points = load_layer_points(DATA_PATH)
    print(f'检测到层: {list(layer_points.keys())}')
    xi, yi, grid_points = build_unified_grid(layer_points)
    order, z_list = interpolate_all_layers(layer_points, grid_points)

    # 排除最顶层地表层
    layer_names = [name for name in order if name != '地表层']
    print('层插值顺序(自下而上):', layer_names)

    build_block_model(grid_points, z_list, layer_names)

    # 读取钻孔位置数据
    borehole_locations = load_borehole_locations(LOCATION_PATH)

    # 显示所有插值结果图，并标注钻孔位置和名称
    plot_kriging_results(z_list, xi, yi, order, layer_points, borehole_locations)

if __name__ == '__main__':
    main()
