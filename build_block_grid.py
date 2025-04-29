import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def generate_regular_grid(nx=10, ny=8, x_range=(0, 100), y_range=(0, 80)):
    xs = np.linspace(*x_range, nx)
    ys = np.linspace(*y_range, ny)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    return xy, nx, ny

def generate_layer_z(x, y, kind='upper'):
    base = 100 - 0.05 * x + 0.03 * y + 2 * np.sin(x / 15)
    if kind == 'lower':
        return base - 10 - 1.5 * np.sin(y / 10)
    return base

def build_regular_prism_blocks(xy, nx, ny):
    """
    将规则网格点转换为三棱柱体块（每个网格单元 → 两个三棱柱）
    """
    x = xy[:, 0].reshape((ny, nx))
    y = xy[:, 1].reshape((ny, nx))
    upper_z = generate_layer_z(x, y, kind='upper')
    lower_z = generate_layer_z(x, y, kind='lower')

    blocks = []

    for i in range(ny - 1):
        for j in range(nx - 1):
            # 网格单元的 4 个角点索引
            p0 = (i, j)
            p1 = (i, j + 1)
            p2 = (i + 1, j)
            p3 = (i + 1, j + 1)

            # 每个点的上层和下层坐标
            A = [x[p0], y[p0], upper_z[p0]]
            B = [x[p1], y[p1], upper_z[p1]]
            C = [x[p3], y[p3], upper_z[p3]]
            D = [x[p2], y[p2], upper_z[p2]]

            A_ = [x[p0], y[p0], lower_z[p0]]
            B_ = [x[p1], y[p1], lower_z[p1]]
            C_ = [x[p3], y[p3], lower_z[p3]]
            D_ = [x[p2], y[p2], lower_z[p2]]

            # 第一个三棱柱：A-B-C, A_-B_-C_
            block1 = [A, B, C, A_, B_, C_]
            # 第二个三棱柱：A-C-D, A_-C_-D_
            block2 = [A, C, D, A_, C_, D_]

            blocks.append(block1)
            blocks.append(block2)

    return blocks

def plot_blocks(blocks, sample_count=50):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    all_faces = []

    for i, block in enumerate(blocks[:sample_count]):
        A, B, C, A_, B_, C_ = block
        faces = [
            [A, B, C], [A_, B_, C_],
            [A, B, B_], [A, A_, B_],
            [B, C, C_], [B, B_, C_],
            [C, A, A_], [C, C_, A_]
        ]
        all_faces.extend(faces)

    poly = Poly3DCollection(all_faces, facecolor='peachpuff', edgecolor='k', alpha=0.9)
    ax.add_collection3d(poly)

    all_pts = np.vstack([np.array(block) for block in blocks[:sample_count]])
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    ax.set_box_aspect([1, 1, 0.4])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"规则网格构建完整三维岩体块体（共绘制 {sample_count} 个）")
    plt.tight_layout()
    plt.show()

def main():
    xy, nx, ny = generate_regular_grid(nx=20, ny=15)
    blocks = build_regular_prism_blocks(xy, nx, ny)
    plot_blocks(blocks, sample_count=150)  # 可调整数量

if __name__ == "__main__":
    main()
