import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_irregular_points(n_points=50, x_range=(0, 100), y_range=(0, 80), seed=42):
    """
    生成一组在 x-y 平面上不规则分布的三维地层边界点。
    z 值通过构造函数 + 噪声模拟起伏的地层面。
    """
    np.random.seed(seed)
    x = np.random.uniform(*x_range, n_points)
    y = np.random.uniform(*y_range, n_points)

    # 构造模拟地层面：整体倾斜 + 起伏 + 噪声
    z = 100 - 0.1 * x + 0.05 * y + 2 * np.sin(x / 10) + np.random.normal(0, 0.5, n_points)

    return np.vstack((x, y, z)).T

def build_layer_surface(points_3d):
    """
    使用 Delaunay 对三维点的 x-y 平面进行三角剖分
    返回：三角网索引 + 原始点
    """
    points_2d = points_3d[:, :2]
    tri = Delaunay(points_2d)
    return tri.simplices, points_3d

def plot_layer_surface(triangles, points_3d, color='lightblue', alpha=0.8):
    """
    绘制一个三维地层面三角网
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    mesh = [points_3d[tri] for tri in triangles]
    poly = Poly3DCollection(mesh, facecolor=color, edgecolor='gray', alpha=alpha)
    ax.add_collection3d(poly)

    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c='red', s=10, label='控制点')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("不规则地层面三角网建模示意")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    points_3d = generate_irregular_points(n_points=60)  # 可修改数量
    triangles, pts = build_layer_surface(points_3d)
    plot_layer_surface(triangles, pts)

if __name__ == "__main__":
    main()
