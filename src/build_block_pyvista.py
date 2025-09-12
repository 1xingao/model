import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
try:
    import trimesh
except ImportError:
    trimesh = None

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

    def export_to_gltf_trimesh(self, output_path="model.gltf"):
        """
        使用Trimesh导出为GLTF格式，支持材质和颜色
        参数:
            output_path: 导出的文件路径
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
                vertices = mesh.points
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

if __name__ == "__main__":
    builder = Block()
    builder.main()
