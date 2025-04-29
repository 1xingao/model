import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams['font.sans-serif'] = ['SimHei']
def generate_irregular_xy(n_points=120, x_range=(0, 500), y_range=(0, 500), seed=42):
    np.random.seed(seed)
    x = np.random.uniform(*x_range, n_points)
    y = np.random.uniform(*y_range, n_points)
    return np.column_stack((x, y))

def generate_layers_from_xy(xy):
    x, y = xy[:, 0], xy[:, 1]

    z_top = 100 - 0.05 * x + 0.03 * y + 2 * np.sin(x / 15)
    z_mid = z_top - 10 - 5 * np.sin(y / 10)
    z_bot = z_top - 20 - 5 * np.cos(x / 20)

    upper = np.column_stack((x, y, z_top))
    middle = np.column_stack((x, y, z_mid))
    lower = np.column_stack((x, y, z_bot))
    return upper, middle, lower

def build_prism_blocks(upper, lower):
    tri = Delaunay(upper[:, :2])
    simplices = tri.simplices

    blocks = []
    for tri_ids in simplices:
        A, B, C = upper[tri_ids]
        A_, B_, C_ = lower[tri_ids]
        block = [A, B, C, A_, B_, C_]
        blocks.append(block)
    return blocks, simplices

def build_poly_faces_from_block(block):
    A, B, C, A_, B_, C_ = block
    return [
        [A, B, C],       # 上底面
        [A_, B_, C_],    # 下底面
        [A,B,B_,A_],
        [B,C,C_,B_],
        [C,A,A_,C_],
    ]

def plot_three_layer_blocks(blocks1, blocks2, alpha=1):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    faces_upper = []
    for block in blocks1:
        faces_upper.extend(build_poly_faces_from_block(block))

    faces_lower = []
    for block in blocks2:
        faces_lower.extend(build_poly_faces_from_block(block))

    poly1 = Poly3DCollection(faces_upper, facecolor='lightskyblue', edgecolor='gray', alpha=alpha, label='上层块体')
    poly2 = Poly3DCollection(faces_lower, facecolor='lightcoral', edgecolor='gray', alpha=alpha, label='下层块体')

    ax.add_collection3d(poly2)  # 先画下层
    ax.add_collection3d(poly1)  # 后画上层，避免完全遮挡

    # 自动缩放
    all_pts = np.vstack([np.array(block) for block in (blocks1 + blocks2)])
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    ax.set_box_aspect([1, 1, 0.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("三层地层模型（两层块体可视化）")
    plt.tight_layout()
    plt.show()

def main():
    xy = generate_irregular_xy(n_points=300)
    upper, middle, lower = generate_layers_from_xy(xy)

    # 使用上层剖分，所有层共用索引
    blocks_upper, simplices = build_prism_blocks(upper, middle)
    blocks_lower, _ = build_prism_blocks(middle, lower)

    plot_three_layer_blocks(blocks_upper, blocks_lower)

if __name__ == "__main__":
    main()
