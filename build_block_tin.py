import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams['font.sans-serif'] = ['SimHei'] 
def generate_irregular_xy(n_points=100, x_range=(0, 100), y_range=(0, 80), seed=42):

    np.random.seed(seed)
    x = np.random.uniform(*x_range, n_points)
    y = np.random.uniform(*y_range, n_points)
    return np.column_stack((x, y))

def generate_layers_from_xy(xy):

    x, y = xy[:, 0], xy[:, 1]
    z_upper = 100 - 0.05 * x + 0.03 * y + 2 * np.sin(x / 15)
    z_lower = z_upper - 20 - 5 * np.sin(y / 10)
    

    upper = np.column_stack((x, y, z_upper))
    lower = np.column_stack((x, y, z_lower))
    return upper, lower

def build_prism_blocks_from_delaunay(upper, lower):

    tri = Delaunay(upper[:, :2])
    simplices = tri.simplices

    blocks = []
    for tri_ids in simplices:
        A, B, C = upper[tri_ids]
        A_, B_, C_ = lower[tri_ids]
        block = [A, B, C, A_, B_, C_]
        blocks.append(block)
    return blocks

def plot_prism_blocks(blocks):

    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    all_faces = []

    for block in blocks[:]:
        A, B, C, A_, B_, C_ = block
        # faces = [
        #     [A, B, C], [A_, B_, C_],
        #     [A, B, B_], [A, A_, B_],
        #     [B, C, C_], [B, B_, C_],
        #     [C, A, A_], [C, C_, A_]
        # ]
        faces = [
            [A, B, C], [A_, B_, C_],
            [A,B,B_,A_],
            [B,C,C_,B_],
            [C,A,A_,C_],
        ]
        all_faces.extend(faces)

    poly = Poly3DCollection(all_faces, facecolor='peachpuff', edgecolor='gray', alpha=0.95)
    ax.add_collection3d(poly)

    # 自动缩放
    all_pts = np.vstack([np.array(block) for block in blocks])
    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    ax.set_box_aspect([1, 1, 0.4])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"基于不规则三角网构建三棱柱岩层体块")
    plt.tight_layout()
    plt.show()

def main():
    xy = generate_irregular_xy(n_points=150)
    upper, lower = generate_layers_from_xy(xy)
    blocks = build_prism_blocks_from_delaunay(upper, lower)
    plot_prism_blocks(blocks)

if __name__ == "__main__":
    main()
