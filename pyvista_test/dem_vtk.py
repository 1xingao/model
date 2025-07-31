from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import examples


# Seed random numbers for reproducibility
rng = np.random.default_rng(seed=0)
def dem_plot():
    dem = examples.download_crater_topo()

    subset = dem.extract_subset((500, 900, 400, 800, 0, 0), (5, 5, 1))
    subset.plot(cpos='xy')

    terrain = subset.warp_by_scalar()

    terrain.plot(cpos='xy', show_edges=True, line_width=2)



def triangle_plot():
    # Define a simple Gaussian surface
    n = 20
    x = np.linspace(-200, 200, num=n) + rng.uniform(-5, 5, size=n)
    y = np.linspace(-200, 200, num=n) + rng.uniform(-5, 5, size=n)
    xx, yy = np.meshgrid(x, y)
    A, b = 100, 100
    zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

    # Get the points as a 2D NumPy array (N by 3)
    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    points[0:5, :]

    # simply pass the numpy points to the PolyData constructor
    cloud = pv.PolyData(points)
    cloud.plot(point_size=15)

    surf = cloud.delaunay_2d()
    surf.plot(show_edges=True)

    x = np.arange(10, dtype=float)
    xx, yy, zz = np.meshgrid(x, x, [0])
    points = np.column_stack((xx.ravel(order='F'), yy.ravel(order='F'), zz.ravel(order='F')))
    # Perturb the points
    points[:, 0] += rng.random(len(points)) * 0.3
    points[:, 1] += rng.random(len(points)) * 0.3
    # Create the point cloud mesh to triangulate from the coordinates
    cloud = pv.PolyData(points)

    #边界问题
    # surf = cloud.delaunay_2d()
    # surf.plot(cpos='xy', show_edges=True)

    # surf = cloud.delaunay_2d(alpha=1.0)
    # surf.plot(cpos='xy', show_edges=True)
    # Define a polygonal hole with a clockwise polygon

    
    ids = [22, 23, 24, 25, 35, 45, 44, 43, 42, 32]

    # Create a polydata to store the boundary
    polygon = pv.PolyData()
    # Make sure it has the same points as the mesh being triangulated
    polygon.points = points
    # But only has faces in regions to ignore
    polygon.faces = np.insert(ids, 0, len(ids))

    surf = cloud.delaunay_2d(alpha=1.0, edge_source=polygon)

    p = pv.Plotter()
    p.add_mesh(surf, show_edges=True)
    p.add_mesh(polygon, color='red', opacity=0.5)
    p.show(cpos='xy')

triangle_plot()