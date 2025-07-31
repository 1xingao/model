import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay

class Block:
    def __init__(self, xy=None, z_list=None):
        self.xy = xy
        self.z_list = z_list

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

    def visualization_block(self):
        layer_list = self.generate_layers_from_xyz()
        # 构建两层块体
        block_list = []
        for i in range(len(layer_list)-1):

            blocks1 = self.build_prism_blocks(layer_list[i], layer_list[i+1])
            block_list.append(blocks1)

        mesh_list = []
        # 创建 PyVista 网格
        for i in range(len(block_list)):
            mesh = self.create_pyvista_mesh_from_blocks(block_list[i])
            mesh_list.append(mesh)

        # 可视化
        plotter = pv.Plotter()
        plotter.add_mesh(mesh_list[2], color='lightcoral', opacity=0.9, show_edges=True, label='layer2-Lower')
        plotter.add_mesh(mesh_list[1], color='lightskyblue', opacity=0.9, show_edges=True, label='layer1-layer2')
        plotter.add_mesh(mesh_list[0], color='lightgreen', opacity=0.9, show_edges=True, label='Upper-layer1')

        plotter.add_legend()
        
        plotter.add_axes()
        plotter.show_grid()
        plotter.show(title=f'{len(mesh_list)+1}层地层体块模型(PyVista)')
        
        # mesh_list[-1].save('upper_layer.obj')
        # mesh_list[-2].save('lower_layer.obj')
        # mesh_list[-3].save('middle_layer.obj')

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

if __name__ == "__main__":
    builder = Block()
    builder.main()
