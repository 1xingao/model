import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
import trimesh
import py3dtiles
import matplotlib.font_manager as fm
#设置字体文件路径
font_path = 'SimHei.ttf'
#注册字体文件
prop = fm.FontProperties(fname=font_path)
class Block:
    def __init__(self, xy=None, z_list=None, layer_names=None):
        self.xy = xy
        self.z_list = z_list
        self.layer_names = layer_names  # 添加地层名称
        self.mesh_list = []

    # 实际数据
    def generate_layers_from_xyz(self):#z_list 中的数据是每层相交点的坐标
        x, y = self.xy[:, 0], self.xy[:, 1]
        layer_list = []
        for z in self.z_list:
            layer = np.column_stack((x, y, z))
            layer_list.append(layer)
        
        return layer_list

    def build_prism_blocks(self,upper, lower):
        tri = Delaunay(upper[:, :2])
        simplices = tri.simplices
        blocks = []
        for tri_ids in simplices:
            A, B, C = upper[tri_ids]
            A_, B_, C_ = lower[tri_ids]
            blocks.append([A, B, C, A_, B_, C_])
        return blocks

    def create_pyvista_mesh_from_blocks(self,blocks):
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


    def execute(self):
        self.visualization_block()

    def visualization_block(self, screenshot_path=None, title=None, off_screen=False):
        """可视化当前 Block 的层间块体。
        screenshot_path: 保存截图
        title: 窗口标题
        off_screen: 在无界面环境使用离屏渲染
        """
        layer_list = self.generate_layers_from_xyz()
        if len(layer_list) < 2:
            raise ValueError("需要至少两层数据才能构建块体")

        # 构建相邻层之间的三棱柱块集合
        # block_list = [self.build_prism_blocks(layer_list[i], layer_list[i+1])
        #               for i in range(len(layer_list)-1)]
        block_list = []
        cnt = 0
        interval = 0
        for i in range(len(layer_list)-1):
        
            blocks1 = self.build_prism_blocks(layer_list[i]+np.array([0,0,cnt]), layer_list[i+1]+np.array([0,0,cnt]))
            block_list.append(blocks1)
            cnt += interval

        mesh_list = [self.create_pyvista_mesh_from_blocks(blks) for blks in block_list]
        self.mesh_list = mesh_list  # 保存以便后续导出使用
        # 扩展颜色列表
        extended_colors = [
            'lightgreen', 'lightskyblue', 'lightcoral', 'khaki', 'plum',
            'gold', 'darkorange', 'cyan', 'magenta', 'lime', 'pink'
        ]

        plotter = pv.Plotter(off_screen=off_screen)
        for idx, mesh in enumerate(mesh_list[::-1]):  # 反向绘制保证上层不被完全遮挡
            color = extended_colors[idx % len(extended_colors)]
            layer_label = self.layer_names[len(mesh_list) - idx - 1] if self.layer_names else f'layer{len(mesh_list)-idx}'
            plotter.add_mesh(mesh, color=color, opacity=1, show_edges=True, label=layer_label)
        
        # plotter.set_scale(zscale=10)
        plotter.add_legend()
        plotter.add_axes()
        plotter.show_grid(color='black') 

        window_title = title or f'{len(mesh_list)+1}层地层体块模型(PyVista)'
        if screenshot_path:
            plotter.show(title=window_title, screenshot=screenshot_path, auto_close=True)
        else:
            plotter.show(title=window_title)
        

    def split_block(self,main_xy,another_xy):
        layer_list = self.generate_layers_from_xy(main_xy)
        another_layer_list = self.generate_layers_from_xy(another_xy)
        # 构层块体
        block_list = []
        for i in range(len(layer_list)-1):

            blocks1 = self.build_prism_blocks(layer_list[i], layer_list[i+1])
            block_list.append(blocks1)

        another_block_list = []
        for i in range(len(another_layer_list)-1):
            
            blocks1 = self.build_prism_blocks(another_layer_list[i], another_layer_list[i+1])
            another_block_list.append(blocks1)

        mesh_list = []
        # 创建 PyVista 网格
        for i in range(len(block_list)):
            mesh = self.create_pyvista_mesh_from_blocks(block_list[i])
            mesh_list.append(mesh)

        another_mesh_list = []
        # 创建 PyVista unvisable 网格
        for i in range(len(another_block_list)):
            mesh = self.create_pyvista_mesh_from_blocks(another_block_list[i])
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

    def export_model(self, output_path="model.vtm"):
        """
        导出模型为 .vtm 文件，支持不同地层显示不同颜色。
        参数:
            output_path: 导出的文件路径
        """

        # 为每个地层分配颜色
        extended_colors = [
            'lightgreen', 'lightskyblue', 'lightcoral', 'khaki', 'plum',
            'gold', 'darkorange', 'cyan', 'magenta', 'lime', 'pink'
        ]

        combined_mesh = pv.MultiBlock()
        for idx, mesh in enumerate(self.mesh_list[::-1]):
            color = extended_colors[idx % len(extended_colors)]
            layer_label = self.layer_names[len(self.mesh_list) - idx - 1] if self.layer_names else f'layer{len(self.mesh_list)-idx}'
            mesh["layer"] = layer_label.encode('ascii', 'ignore').decode('ascii')  # 确保地层名称为 ASCII 编码
            mesh["color"] = color  # 添加颜色属性
            combined_mesh.append(mesh)

        # 导出为 .vtm 文件
        combined_mesh.save(output_path)
        print(f"模型已导出到 {output_path}")

    def export_to_gltf_trimesh(self, output_path="model.gltf", rotate_axes=True):
        """
        使用Trimesh导出为GLTF格式，支持材质和颜色
        参数:
            output_path: 导出的文件路径
            rotate_axes: 是否调整坐标轴以修复旋转问题 (默认True)
        """
        if trimesh is None:
            print("需要安装trimesh库：pip install trimesh[easy]")
            return
            
        if not self.mesh_list:
            raise ValueError("没有可导出的网格数据，请先执行 visualization_block 方法")
            
        try:
            # 扩展颜色列表 (RGBA格式)
            extended_colors = [
                [144, 238, 144, 255],  # lightgreen
                [135, 206, 250, 255],  # lightskyblue  
                [240, 128, 128, 255],  # lightcoral
                [240, 230, 140, 255],  # khaki
                [221, 160, 221, 255],  # plum
                [255, 215, 0, 255],    # gold
                [255, 140, 0, 255],    # darkorange
                [0, 255, 255, 255],    # cyan
                [255, 0, 255, 255],    # magenta
                [0, 255, 0, 255],      # lime
                [255, 192, 203, 255],  # pink
            ]
            
            scene = trimesh.Scene()
            
            for idx, mesh in enumerate(self.mesh_list):
                # 获取顶点和面数据
                vertices_original = mesh.points
                if rotate_axes:
                    # 修复坐标轴方向：确保Z轴垂直向上，地层垂直排列
                    # 原始: (X, Y, Z) -> 调整: (X, Z, Y) 
                    vertices = np.column_stack((
                        vertices_original[:, 0],  # X保持不变
                        vertices_original[:, 2],  # Z作为新的Y (垂直方向)
                        vertices_original[:, 1]   # Y作为新的Z (深度方向)
                    ))
                else:
                    vertices = vertices_original
                faces_data = mesh.faces
                
                # 处理面数据：PyVista的面数据格式为 [n, v1, v2, v3, ...] 
                # 需要转换为trimesh的三角形面格式
                faces = []
                i = 0
                while i < len(faces_data):
                    n_vertices = faces_data[i]
                    if n_vertices == 3:  # 三角形面
                        faces.append(faces_data[i+1:i+4])
                    elif n_vertices == 4:  # 四边形面，分解为两个三角形
                        quad = faces_data[i+1:i+5]
                        faces.append([quad[0], quad[1], quad[2]])
                        faces.append([quad[0], quad[2], quad[3]])
                    i += n_vertices + 1
                
                if not faces:
                    print(f"警告：第{idx}层网格没有有效的面数据，跳过")
                    continue
                
                faces = np.array(faces)
                
                # 创建trimesh对象
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                # 设置颜色
                color = extended_colors[idx % len(extended_colors)]
                tri_mesh.visual.face_colors = color
                
                # 添加到场景
                layer_name = self.layer_names[idx] if self.layer_names and idx < len(self.layer_names) else f'layer_{idx}'
                # 确保层名称为ASCII编码
                layer_name_ascii = layer_name.encode('ascii', 'ignore').decode('ascii')
                scene.add_geometry(tri_mesh, node_name=layer_name_ascii)
            
            # 导出为GLTF
            scene.export(output_path)
            print(f"GLTF模型已导出到 {output_path}")
            
        except Exception as e:
            print(f"导出GLTF时出错: {e}")
            print("请确保已安装完整的trimesh库：pip install trimesh[easy]")

    def export_to_3dtiles(self, output_dir="3dtiles_model", center_coords=None, rotate_axes=True):
        """
        导出模型为3DTiles格式，适用于Cesium等Web 3D应用
        参数:
            output_dir: 导出的目录路径
            center_coords: 模型中心坐标 [longitude, latitude, height]，默认为 [0, 0, 0]
            rotate_axes: 是否调整坐标轴以修复旋转问题 (默认True)
        """
        if py3dtiles is None:
            print("需要安装py3dtiles库：pip install py3dtiles")
            return
            
        if not self.mesh_list:
            raise ValueError("没有可导出的网格数据，请先执行 visualization_block 方法")
            
        try:
            import os
            import json
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置默认中心坐标
            if center_coords is None:
                center_coords = [116.0, 39.0, 0]  # 北京坐标作为默认值
                
            lon, lat, height = center_coords
            
            # 创建tileset.json文件
            tileset = {
                "asset": {"version": "1.0"},
                "geometricError": 500,
                "root": {
                    "boundingVolume": {
                        "region": [
                            np.radians(lon - 0.01),  # west
                            np.radians(lat - 0.01),  # south  
                            np.radians(lon + 0.01),  # east
                            np.radians(lat + 0.01),  # north
                            height - 100,            # minimum height
                            height + 100             # maximum height
                        ]
                    },
                    "geometricError": 100,
                    "children": []
                }
            }
            
            # 扩展颜色列表 (RGB格式，0-1范围)
            extended_colors = [
                [0.565, 0.933, 0.565],  # lightgreen
                [0.529, 0.808, 0.980],  # lightskyblue
                [0.941, 0.502, 0.502],  # lightcoral
                [0.941, 0.902, 0.549],  # khaki
                [0.867, 0.627, 0.867],  # plum
                [1.0, 0.843, 0.0],      # gold
                [1.0, 0.549, 0.0],      # darkorange
                [0.0, 1.0, 1.0],        # cyan
                [1.0, 0.0, 1.0],        # magenta
                [0.0, 1.0, 0.0],        # lime
                [1.0, 0.753, 0.796],    # pink
            ]
            
            # 为每个地层创建一个tile
            for idx, mesh in enumerate(self.mesh_list):
                layer_name = self.layer_names[idx] if self.layer_names and idx < len(self.layer_names) else f'layer_{idx}'
                layer_name_ascii = layer_name.encode('ascii', 'ignore').decode('ascii')
                
                # 先导出为GLTF格式
                gltf_filename = f"{layer_name_ascii}_{idx}.gltf"
                gltf_path = os.path.join(output_dir, gltf_filename)
                
                # 使用trimesh导出单个地层为GLTF
                if trimesh is not None:
                    vertices_original = mesh.points
                    if rotate_axes:
                        # 修复坐标轴方向：确保Z轴垂直向上，地层垂直排列
                        vertices = np.column_stack((
                            vertices_original[:, 0],  # X保持不变
                            vertices_original[:, 2],  # Z作为新的Y (垂直方向)
                            vertices_original[:, 1]   # Y作为新的Z (深度方向)
                        ))
                    else:
                        vertices = vertices_original
                    faces_data = mesh.faces
                    
                    # 处理面数据
                    faces = []
                    i = 0
                    while i < len(faces_data):
                        n_vertices = faces_data[i]
                        if n_vertices == 3:
                            faces.append(faces_data[i+1:i+4])
                        elif n_vertices == 4:
                            quad = faces_data[i+1:i+5]
                            faces.append([quad[0], quad[1], quad[2]])
                            faces.append([quad[0], quad[2], quad[3]])
                        i += n_vertices + 1
                    
                    if faces:
                        faces = np.array(faces)
                        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        
                        # 设置颜色
                        color_rgb = extended_colors[idx % len(extended_colors)]
                        color_rgba = color_rgb + [1.0]  # 添加alpha通道
                        tri_mesh.visual.face_colors = [int(c * 255) for c in color_rgba]
                        
                        # 导出为GLTF
                        tri_mesh.export(gltf_path)
                        
                        # 添加到tileset
                        child_tile = {
                            "boundingVolume": {
                                "region": [
                                    np.radians(lon - 0.005),
                                    np.radians(lat - 0.005),
                                    np.radians(lon + 0.005),
                                    np.radians(lat + 0.005),
                                    height + idx * 10 - 50,
                                    height + idx * 10 + 50
                                ]
                            },
                            "geometricError": 50,
                            "content": {
                                "uri": gltf_filename
                            }
                        }
                        tileset["root"]["children"].append(child_tile)
                        
                        print(f"已导出地层 {layer_name} 到 {gltf_filename}")
                else:
                    print(f"警告：trimesh未安装，无法导出地层 {layer_name}")
            
            # 保存tileset.json
            tileset_path = os.path.join(output_dir, "tileset.json")
            with open(tileset_path, 'w', encoding='utf-8') as f:
                json.dump(tileset, f, indent=2, ensure_ascii=False)
            
            print(f"3DTiles模型已导出到 {output_dir}")
            print(f"主文件: {tileset_path}")
            print(f"模型中心坐标: 经度 {lon}, 纬度 {lat}, 高度 {height}")
            print("可以在Cesium等支持3DTiles的应用中加载此模型")
            
        except Exception as e:
            print(f"导出3DTiles时出错: {e}")
            print("请确保已安装py3dtiles库：pip install py3dtiles")

if __name__ == "__main__":
    # 创建示例数据
    # xy = np.array([[0, 0], [1, 0], [0.5, 1], [1.5, 1]])
    # z_list = [np.array([0, 0, 0, 0]), 
    #           np.array([1, 1, 1, 1]), 
    #           np.array([2, 2, 2, 2])]
    # layer_names = ['表土层', '砂土层', '粘土层']
    
    # builder = Block(xy=xy, z_list=z_list, layer_names=layer_names)
    
    # # 生成可视化
    # builder.visualization_block(title='地质体三维模型')
    
    # # 导出VTM格式
    # builder.export_model("output_model.vtm")
    
    # # 导出GLTF格式（使用Trimesh，自动修正90度旋转）
    # builder.export_to_gltf_trimesh("output_model.gltf", rotate_axes=True)
    
    # # 导出3DTiles格式（用于Cesium等Web 3D应用，自动修正90度旋转）
    # # 可以指定地理坐标，例如：[116.0, 39.0, 0] 表示北京
    # builder.export_to_3dtiles("3dtiles_output", center_coords=[116.0, 39.0, 0], rotate_axes=True)
    
    # 请替换为你的实际数据
    print("请使用实际的 xy 坐标和 z_list 数据来创建 Block 实例")
    print("支持的导出格式:")
    print("1. VTM格式: builder.export_model('model.vtm')")
    print("2. GLTF格式: builder.export_to_gltf_trimesh('model.gltf', rotate_axes=True)")  
    print("3. 3DTiles格式: builder.export_to_3dtiles('output_dir', center_coords=[lon, lat, height], rotate_axes=True)")
    print()
    print("关于旋转修正:")
    print("- rotate_axes=True: 修正90度旋转，地层垂直排列（推荐）")
    print("- rotate_axes=False: 保持原始坐标轴，地层可能横向排列")
    print()
    print("示例用法:")
    print("builder = Block(xy=your_xy_data, z_list=your_z_data, layer_names=your_layer_names)")
    print("builder.visualization_block()")
    print("builder.export_to_gltf_trimesh('model.gltf', rotate_axes=True)  # 修正旋转")
    print("builder.export_to_3dtiles('geological_model_3dtiles', rotate_axes=True)  # 修正旋转")
