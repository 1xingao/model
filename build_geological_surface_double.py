import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_base_xy_grid(n_x=10, n_y=8, x_range=(0, 100), y_range=(0, 80), noise=5):
    """
    在给定范围内生成规则的 x-y 网格点，可加微小扰动模拟非规则性
    返回：N x 2 数组，每行是一个 (x, y)
    """
    xs = np.linspace(*x_range, n_x)
    ys = np.linspace(*y_range, n_y)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T

    # 加一点点扰动，模拟实际测点非规则分布
    np.random.seed(42)
    xy += np.random.normal(scale=noise, size=xy.shape)
    return xy

def generate_layer_from_xy(xy, z_func):
    """
    输入一组 (x, y) 坐标和一个 z 生成函数，返回三维点
    """
    z = z_func(xy[:, 0], xy[:, 1])
    return np.column_stack((xy, z))

def build_surface(points_3d):
    """
    使用 Delaunay 对 x-y 平面做三角剖分，返回三角索引
    """
    tri = Delaunay(points_3d[:, :2])
    return tri.simplices

def plot_two_surfaces_and_points(tri, upper_pts, lower_pts):
    """
    同时绘制上下两层地层面（颜色区分），并显示控制点
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 上层面
    poly1 = Poly3DCollection([upper_pts[t] for t in tri], facecolor='skyblue', alpha=0.85, edgecolor='k')
    ax.add_collection3d(poly1)

    # 下层面
    poly2 = Poly3DCollection([lower_pts[t] for t in tri], facecolor='lightgreen', alpha=0.85, edgecolor='k')
    ax.add_collection3d(poly2)

    # 控制点
    ax.scatter(upper_pts[:, 0], upper_pts[:, 1], upper_pts[:, 2], c='b', s=10, label='上层点')
    ax.scatter(lower_pts[:, 0], lower_pts[:, 1], lower_pts[:, 2], c='g', s=10, label='下层点')

    # for p1, p2 in zip(upper_pts, lower_pts):
    #     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("上下两层地层面（三维点对齐）")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    # 1. 构建基础 xy 网格
    xy = generate_base_xy_grid(n_x=12, n_y=10)

    # 2. 定义上下两层的 z 值函数
    def z_upper(x, y):
        return 100 - 0.05 * x + 0.03 * y + 2 * np.sin(x / 15)

    def z_lower(x, y):
        return z_upper(x, y) - 15 - 5 * np.sin(y / 10)  # 比上层低 10～12 米不等

    # 3. 构造两层三维点（具有相同的 x, y）
    upper_layer = generate_layer_from_xy(xy, z_upper)
    lower_layer = generate_layer_from_xy(xy, z_lower)

    # 4. 使用上层的点构建三角网（三角面拓扑共用）
    triangles = build_surface(upper_layer)

    # 5. 可视化
    plot_two_surfaces_and_points(triangles, upper_layer, lower_layer)

if __name__ == "__main__":
    main()
