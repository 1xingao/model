from pyvista import examples
import pyvista as pv
import numpy as np

# 基本绘制
# mesh = examples.download_bunny_coarse()

# pl = pv.Plotter()
# pl.add_mesh(mesh, show_edges=True, color='white')
# pl.add_points(mesh.points, color='red',
#               point_size=2)
# pl.camera_position = [(0.02, 0.30, 0.73),
#                       (0.02, 0.03, -0.022),
#                       (-0.03, 0.94, -0.34)]
# pl.show()


mesh = examples.load_hexbeam()

pl = pv.Plotter()
pl.add_mesh(mesh, show_edges=True, color='white')
pl.add_points(mesh.points, color='red', point_size=20)

single_cell = mesh.extract_cells(mesh.n_cells - 1)
pl.add_mesh(single_cell, color='pink', edge_color='blue',
            line_width=5, show_edges=True)

pl.camera_position = [(6.20, 3.00, 7.50),
                      (0.16, 0.13, 2.65),
                      (-0.28, 0.94, -0.21)]
pl.show()

mesh.point_data['my point values'] = np.arange(mesh.n_points)
mesh.plot(scalars='my point values', cpos=pl.camera_position, show_edges=True)

mesh.cell_data['my cell values'] = np.arange(mesh.n_cells)
mesh.plot(scalars='my cell values', cpos=pl.camera_position, show_edges=True)