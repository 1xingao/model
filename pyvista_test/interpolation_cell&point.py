import pyvista as pv
from pyvista import examples
import vtk

def main():
    uni = examples.load_uniform()
    # reuse the camera position from the previous plotter
    # cpos = pl.camera_position
    # 演示点属性和单元属性的插值效果

    pl = pv.Plotter(shape=(1, 2), border=False)
    pl.add_mesh(uni, scalars='Spatial Point Data', show_edges=True)
    pl.subplot(0, 1)
    pl.add_mesh(uni, scalars='Spatial Cell Data', show_edges=True)
    pl.show()


    cube = pv.Cube()
    cube.cell_data['myscalars'] = range(6)

    other_cube = cube.copy()
    other_cube.point_data['myscalars'] = range(8)

    pl = pv.Plotter(shape=(1, 2), border_width=1)
    pl.add_mesh(cube, cmap='coolwarm')
    pl.subplot(0, 1)
    pl.add_mesh(other_cube, cmap='coolwarm')
    pl.show()


def note():



    # vtk直接转pyvista
    # vtk创建数组的方式，更推荐使用numpy
    vtk_array = vtk.vtkDoubleArray()
    vtk_array.SetNumberOfComponents(3)
    vtk_array.SetNumberOfValues(9)
    vtk_array.SetValue(0, 0)
    vtk_array.SetValue(1, 0)
    vtk_array.SetValue(2, 0)
    vtk_array.SetValue(3, 1)
    vtk_array.SetValue(4, 0)
    vtk_array.SetValue(5, 0)
    vtk_array.SetValue(6, 0.5)
    vtk_array.SetValue(7, 0.667)
    vtk_array.SetValue(8, 0)
    print(vtk_array)

    wrapped = pv.wrap(vtk_array)
    wrapped


main()