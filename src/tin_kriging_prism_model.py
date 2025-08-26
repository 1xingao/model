import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from build_block_pyvista import Block

DATA_PATH = './data/地层坐标.xlsx'  # 输入数据: layer,x,y,z 
GRID_NX = 80
GRID_NY = 80
DEFAULT_VARIOGRAM = 'spherical'            # 默认变差函数模型
LAYER_VARIOGRAM = {}                    # 可为特定层指定: {'Topsoil':'spherical','Coal':'linear'}
VERBOSE_KRIGE = False
layer_name__list = ["sandstone_3","coal5-3","sandstone_2","coal5-2","sandstone_1",
                    "coal4-2","Sandstone and mudstone mixed layer","coal3-1",
                    "gravel sandstone layer","loose layer"]
def load_layer_points(path: str):
    """
    读取地层坐标数据，文件格式为地层名称、x、y、z。
    参数:
        path: 文件路径
    返回:
        字典，其中键是地层名称，值是包含 x, y, z 的 DataFrame。
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
    return grid_points

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

def krige_layer(df_layer: pd.DataFrame, grid_points: np.ndarray, variogram_model: str):
    if len(df_layer) < 3:
        raise ValueError('点数不足，无法克里金插值 (>=3)')
    ok = OrdinaryKriging(
        df_layer['x'], df_layer['y'], df_layer['z']*10,
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
        z_vals = krige_layer(layer_points[lname], grid_points, model)
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
    block.export_model("./data/output_model_random.vtm")

def main():
    print(f'读取数据: {DATA_PATH}')
    layer_points = load_layer_points(DATA_PATH)
    print(f'检测到层: {list(layer_points.keys())}')
    #grid_points = build_unified_grid(layer_points)
    grid_points = build_random_grid(layer_points, num_points=GRID_NX * GRID_NY)
    order, z_list = interpolate_all_layers(layer_points, grid_points)

    # 排除最顶层地表层
    layer_names = [name for name in order if name != '地表层']
    print('层插值顺序(自下而上):', layer_names)

    build_block_model(grid_points, z_list, layer_names)

if __name__ == '__main__':
    main()
