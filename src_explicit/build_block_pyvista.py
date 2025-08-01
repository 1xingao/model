import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay

# 测试数据
def generate_irregular_xy(n_points=4000, x_range=(0, 1500), y_range=(0, 1500), seed=42):
    np.random.seed(seed)
    x = np.random.uniform(*x_range, n_points)
    y = np.random.uniform(*y_range, n_points)
    return np.column_stack((x, y))

def generate_layers_from_xy(xy):
    x, y = xy[:, 0], xy[:, 1]
    # z_top = 100 - 0.05 * x + 0.03 * y + 2 * np.sin(x / 5)
    # z_layer1 = z_top - 100 - 5 * np.sin(y*5)
    # z_layer2 = z_layer1 - 100 - 5 * np.cos(x *5)
    # z_bot = z_layer2 - 100 - 5 * np.cos(x *5)

    # for i in range(len(z_top)):
    #     if x[i] <300 and y[i]<300:
    #         z_layer2[i] = z_bot[i]
    #     if x[i] >700 and y[i] >700:
    #         z_layer2[i] = z_bot[i]

# 塌陷中心和控制参数
    center_x, center_y = 750, 750
    collapse_radius = 200  # 半径更小 -> 中心更陡
    collapse_depth = 150    # 深度更大 -> 更凹陷

    # 到中心的欧几里得距离
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # 更陡峭的高斯塌陷（越靠近中心越剧烈）
    collapse = -collapse_depth * np.exp(-(r ** 2) / (2 * collapse_radius ** 2))

    # 顶板：地形起伏 + 强烈塌陷
    z_top = (
        130
        + 10 * np.sin(x / 80)
        + 8 * np.cos(y / 70)
        + 12 * np.sin(np.sqrt(x ** 2 + y ** 2) / 100)
        + collapse  # 明显塌陷
    )

    # 中间层层与地表相对接近，但逐层弱化塌陷
    z_layer1 = (
        z_top
        - 90
        - 5 * np.sin(y / 30)
        - 4 * np.cos(x / 40)
        + collapse * 0.3  # 继承部分塌陷趋势
    )

    z_layer2 = (
        z_layer1
        - 90
        - 6 * np.sin((x + y) / 25)
        + 5 * np.cos((x - y) / 20)
        + collapse * 0.1
    )

    z_bot = (
        z_layer2
        - 100
        - 12 * np.sin(x / 15)
        - 8 * np.cos(y / 12)
    )


    upper = np.column_stack((x, y, z_top))
    layer_1 = np.column_stack((x, y, z_layer1))
    layer_2 = np.column_stack((x, y, z_layer2))
    lower = np.column_stack((x, y, z_bot))
    return [upper, layer_1, layer_2, lower]


def build_prism_blocks(upper, lower):
    tri = Delaunay(upper[:, :2])
    simplices = tri.simplices
    blocks = []
    for tri_ids in simplices:
        A, B, C = upper[tri_ids]
        A_, B_, C_ = lower[tri_ids]
        blocks.append([A, B, C, A_, B_, C_])
    return blocks

def create_pyvista_mesh_from_blocks(blocks):
    all_faces = []
    all_points = []
    point_id_map = {}
    index = 0

    def get_point_id(p):
        nonlocal index
        key = tuple(np.round(p, 6))
        if key not in point_id_map:
            point_id_map[key] = index
            all_points.append(key)
            index += 1
        return point_id_map[key]

    for block in blocks:
        A, B, C, A_, B_, C_ = block
        pts = [A, B, C, A_, B_, C_]
        ids = [get_point_id(p) for p in pts]

        # 构建面：每个面由点数 + 点索引构成
        # 上面 ABC，下面 A′B′C′，侧面 3 个面
        faces = [
            [3, ids[0], ids[1], ids[2]],  # top
            [3, ids[3], ids[4], ids[5]],  # bottom
            # [3, ids[0], ids[1], ids[4]],
            # [3, ids[0], ids[4], ids[3]],
            # [3, ids[1], ids[2], ids[5]],
            # [3, ids[1], ids[5], ids[4]],
            # [3, ids[2], ids[0], ids[3]],
            # [3, ids[2], ids[3], ids[5]],
            [4, ids[0], ids[1], ids[4], ids[3]],  # side 1
            [4, ids[1], ids[2], ids[5], ids[4]],  # side 2
            [4, ids[2], ids[0], ids[3], ids[5]],  # side 3
        ]
        all_faces.extend(faces)


    # 构建 pyvista mesh
    faces_flat = np.hstack(all_faces)
    mesh = pv.PolyData(np.array(all_points), faces_flat)
    mesh.clean(inplace=True)
    return mesh


def main():
    # 数据生成
    xy = generate_irregular_xy()
    # main_xy = xy[(xy[:,0]<500)|(xy[:,1]<500)]
    # another_xy = xy[(xy[:,0]>500)&(xy[:,1]>500)]
    # split_block(main_xy,another_xy)
    visualization_block(xy)


def visualization_block(xy):
    layer_list = generate_layers_from_xy(xy)
    # 构建两层块体
    #show_single_tin(layer_list)
    #show_single_tin_surface(layer_list)
    block_list = []
    cnt = 0
    for i in range(len(layer_list)-1):
        
        blocks1 = build_prism_blocks(layer_list[i]+-np.array([0,0,cnt]), layer_list[i+1]-np.array([0,0,cnt]))
        block_list.append(blocks1)
        cnt += 200
    mesh_list = []
    # 创建 PyVista 网格
    for i in range(len(block_list)):
        mesh = create_pyvista_mesh_from_blocks(block_list[i])
        mesh_list.append(mesh)
    show_full_model_colormap(mesh_list)
    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_list[2], color='lightcoral', opacity=1, show_edges=False, label='layer2-Lower')
    plotter.add_mesh(mesh_list[1], color='lightskyblue', opacity=1, show_edges=False, label='layer1-layer2')
    plotter.add_mesh(mesh_list[0], color='lightgreen', opacity=1, show_edges=True, label='Upper-layer1')

    plotter.add_legend()
    
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title=f'{len(mesh_list)+1}层地层体块模型(PyVista)')
    
    # mesh_list[-1].save('upper_layer.obj')
    # mesh_list[-2].save('lower_layer.obj')
    # mesh_list[-3].save('middle_layer.obj')

def split_block(main_xy,another_xy):
    layer_list = generate_layers_from_xy(main_xy)
    another_layer_list = generate_layers_from_xy(another_xy)
    # 构层块体
    block_list = []
    for i in range(len(layer_list)-1):
        
        blocks1 = build_prism_blocks(layer_list[i], layer_list[i+1])
        block_list.append(blocks1)

    another_block_list = []
    for i in range(len(another_layer_list)-1):
        
        blocks1 = build_prism_blocks(another_layer_list[i], another_layer_list[i+1])
        another_block_list.append(blocks1)

    mesh_list = []
    # 创建 PyVista 网格
    for i in range(len(block_list)):
        mesh = create_pyvista_mesh_from_blocks(block_list[i])
        mesh_list.append(mesh)

    another_mesh_list = []
    # 创建 PyVista unvisable 网格
    for i in range(len(another_block_list)):
        mesh = create_pyvista_mesh_from_blocks(another_block_list[i])
        another_mesh_list.append(mesh)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_list[2], color='lightcoral', opacity=1, show_edges=True, label='layer2-Lower')
    plotter.add_mesh(mesh_list[1], color='lightskyblue', opacity=1, show_edges=True, label='layer1-layer2')
    plotter.add_mesh(mesh_list[0], color='lightgreen', opacity=1, show_edges=True, label='Upper-layer1')

    plotter.add_mesh(another_mesh_list[2], color='lightcoral', opacity=0.02,  label='layer2-Lower_un')
    plotter.add_mesh(another_mesh_list[1], color='lightskyblue', opacity=0.02, label='layer1-layer2_un')
    plotter.add_mesh(another_mesh_list[0], color='lightgreen', opacity=0.02,  label='Upper-layer1_un')
    plotter.add_legend()
    
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title=f'{len(mesh_list)+1}层地层体块模型(PyVista)')
    
    # mesh_list[-1].save('upper_layer.obj')
    # mesh_list[-2].save('lower_layer.obj')
    # mesh_list[-3].save('middle_layer.obj')

def show_single_tin(layer_list):
    plotter = pv.Plotter()
    for i, layer in enumerate(layer_list):
        # 根据高度着色
        z = layer[:, 2]
        cmap = 'viridis'  # 可选 colormap
        surf = pv.PolyData(layer)
        plotter.add_mesh(surf, scalars=z, cmap=cmap, show_edges=True, label=f'TIN_{i+1}')
    plotter.add_legend()
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title='单独展示每一个TIN（高度着色）')

def show_single_tin_surface(layer_list):
    plotter = pv.Plotter()
    cmap_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    for i, layer in enumerate(layer_list):
        tri = Delaunay(layer[:, :2])
        faces = np.hstack((np.full((tri.simplices.shape[0], 1), 3), tri.simplices)).astype(np.int32)
        surf = pv.PolyData(layer, faces)
        z = layer[:, 2]
        cmap = cmap_list[i % len(cmap_list)]
        plotter.add_mesh(surf, scalars=z, cmap=cmap, show_edges=False, label=f'TIN_{i+1}')
    plotter.add_legend()
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title='单独展示每一个TIN顶面（三角剖分+高度着色）')

def show_full_model_colormap(mesh_list):
    plotter = pv.Plotter()
    # 合并所有 mesh 的点，获取整体高度范围
    all_z = np.concatenate([mesh.points[:, 2] for mesh in mesh_list])
    z_min, z_max = all_z.min(), all_z.max()
    cmap = 'viridis'  # 可选色带
    for i, mesh in enumerate(mesh_list):
        # 按整体高度范围着色
        plotter.add_mesh(mesh, scalars=mesh.points[:, 2], cmap=cmap, clim=[z_min, z_max], show_edges=False, label=f'Block_{i+1}')
    plotter.add_legend()
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title='整体三维模型统一色带（按高度着色）')

# 使用方法：show_full_model_colormap(mesh_list)

if __name__ == "__main__":
    main()
