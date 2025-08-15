import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from build_block_pyvista import Block

# ================= 可调参数接口 =================
CSV_PATH = './data/test_layers_points.csv'  # 输入数据: layer,x,y,z
GRID_NX = 80
GRID_NY = 80
DEFAULT_VARIOGRAM = 'spherical'            # 缺省变差函数模型
LAYER_VARIOGRAM = {}                    # 可为特定层指定: {'Topsoil':'spherical','Coal':'linear'}
VERBOSE_KRIGE = False

def load_layer_points(csv_path: str):
    df = pd.read_csv(csv_path)
    # 统一列名到小写
    rename_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename_map)
    # 兼容可能的大小写 / 额外空白
    required = {'layer','x','y','z'}
    if not required.issubset(df.columns):
        raise ValueError(f'CSV 缺少列: 需要 {required}, 实际 {df.columns.tolist()}')
    # 丢弃包含 NaN 的关键行
    df = df.dropna(subset=['layer','x','y','z'])
    groups = {lname: g[['x','y','z']].reset_index(drop=True) for lname, g in df.groupby('layer')}
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
    xi = np.linspace(x_min, x_max, GRID_NX)
    yi = np.linspace(y_min, y_max, GRID_NY)
    gx, gy = np.meshgrid(xi, yi)
    grid_points = np.c_[gx.ravel(), gy.ravel()]
    return xi, yi, grid_points

def krige_layer(df_layer: pd.DataFrame, grid_points: np.ndarray, variogram_model: str):
    if len(df_layer) < 3:
        raise ValueError('点数不足，无法克里金插值 (>=3)')
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
        z_vals = krige_layer(layer_points[lname], grid_points, model)
        z_list.append(z_vals)
    return order, z_list

def build_block_model(grid_points: np.ndarray, z_list: list):
    block = Block(xy=grid_points, z_list=z_list)
    block.execute()

def main():
    print(f'读取数据: {CSV_PATH}')
    layer_points = load_layer_points(CSV_PATH)
    print(f'检测到层: {list(layer_points.keys())}')
    xi, yi, grid_points = build_unified_grid(layer_points)
    order, z_list = interpolate_all_layers(layer_points, grid_points)
    print('层插值顺序(自下而上):', order)
    build_block_model(grid_points, z_list)

if __name__ == '__main__':
    main()
