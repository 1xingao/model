#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fault_tin_demo.py

演示：生成 Z 字形断层、按断层分割点集、为两侧独立做 2D Delaunay（TIN）、
生成断层三角带（ribbon），保存示例 CSV，并绘制 3D 可视化图（PNG）。

依赖:
    numpy, pandas, scipy, matplotlib
安装:
    pip install numpy pandas scipy matplotlib
运行:
    python fault_tin_demo.py
输出:
    ./output/
      - points_sample.csv
      - footwall_tri_sample.csv
      - Bedrock_subsided.csv
      - Coal_subsided.csv
      - Topsoil_subsided.csv
      - tin_fault_visualization.png
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# -------------------------- 参数与网格生成 --------------------------
OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)

np.random.seed(2)

# XY 域与分辨率（可按需调整）
nx, ny = 30, 32
xs = np.linspace(0, 600, nx)
ys = np.linspace(0, 320, ny)
XX, YY = np.meshgrid(xs, ys)
XY = np.column_stack([XX.ravel(), YY.ravel()])  # (N,2)

# 定义三层基础高度场函数（简单示例）
def layer_base_z(xy, layer_index):
    x, y = xy[:,0], xy[:,1]
    return (10.0 * layer_index) + 2.0 * x + 1.5 * y

Z0 = layer_base_z(XY, 0)  # Bedrock
Z1 = layer_base_z(XY, 1)  # Coal
Z2 = layer_base_z(XY, 2)  # Topsoil

layers = {
    "Bedrock": Z0,
    "Coal": Z1,
    "Topsoil": Z2
}

# -------------------------- 定义 Z 字形断层中轴线与落差分布 --------------------------
fault_xy = np.array([
    [120, 320],
    [200, 210],
    [300, 240],
    [380, 160],
    [520, 80]
], dtype=float)

# 计算顶点的累积参数 t (0..1)
seg_lengths = np.sqrt(np.sum(np.diff(fault_xy, axis=0)**2, axis=1))
cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
total_len = cumlen[-1]
t_verts = cumlen / total_len if total_len > 0 else np.linspace(0,1, fault_xy.shape[0])

# 定义沿断层变化的 throw（落差）函数（示例）
throw_amplitude = 6.0
throw_profile = lambda t: throw_amplitude * (0.6*np.sin(2*np.pi*t) + 0.4*np.cos(np.pi*t))
throws_at_vertices = throw_profile(t_verts)
throw_interp = interp1d(t_verts, throws_at_vertices, kind='cubic', fill_value="extrapolate")

# -------------------------- 几何工具函数 --------------------------
def nearest_point_on_segment(p, a, b):
    """返回点 p 到线段 ab 的最近点 proj 与参数 u (0..1)。"""
    ap = p - a
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 == 0:
        return a.copy(), 0.0
    u = np.dot(ap, ab) / ab_len2
    u_clamped = np.clip(u, 0.0, 1.0)
    proj = a + u_clamped * ab
    return proj, u_clamped

def classify_point_side_and_local_param(p, polyline):
    """
    对点 p (2,)：
    - 找到最近的多段线线段（返回段索引）
    - 返回投影点、沿 polyline 的参数 t(0..1)、侧别 sign (+1 left / -1 right / 1 on-line)
    - 返回距离
    """
    best_d2 = 1e30
    best_seg = 0
    best_proj = None
    best_u = 0.0
    cum = np.concatenate([[0.0], np.cumsum(np.sqrt(np.sum(np.diff(polyline, axis=0)**2, axis=1)))])
    total = cum[-1] if cum[-1] > 0 else 1.0
    for i in range(len(polyline)-1):
        a = polyline[i]
        b = polyline[i+1]
        proj, u = nearest_point_on_segment(p, a, b)
        d2 = np.sum((p-proj)**2)
        if d2 < best_d2:
            best_d2 = d2
            best_seg = i
            best_proj = proj
            best_u = u
            seg_start_len = cum[i]
    proj_len = seg_start_len + best_u * np.linalg.norm(polyline[best_seg+1] - polyline[best_seg])
    t = proj_len / total
    a = polyline[best_seg]
    b = polyline[best_seg+1]
    seg_vec = b - a
    v = p - best_proj
    cross_z = seg_vec[0]*v[1] - seg_vec[1]*v[0]
    side = np.sign(cross_z)
    if side == 0:
        side = 1.0
    return best_seg, best_proj, t, side, np.sqrt(best_d2)

# -------------------------- 按断层侧别对点集分类并施加落差 --------------------------
N = XY.shape[0]
t_vals = np.zeros(N)
side_vals = np.zeros(N)
dist_vals = np.zeros(N)

for i, p in enumerate(XY):
    _, proj, t, side, d = classify_point_side_and_local_param(p, fault_xy)
    t_vals[i] = t
    side_vals[i] = side
    dist_vals[i] = d

throws_points = throw_interp(t_vals)
mask_hanging = side_vals > 0.0  # 定义：正号为悬挂盘（示例）

# 对每层应用落差：悬挂盘按距离衰减施加 throw（指数衰减）
layers_offset = {}
scale = 30.0  # 衰减尺度（米），可调
taper = np.exp(-dist_vals / scale)
for name, Z in layers.items():
    Z_off = Z.copy()
    Z_off[mask_hanging] += throws_points[mask_hanging] * taper[mask_hanging]
    layers_offset[name] = Z_off

# -------------------------- 分侧点集并做 2D Delaunay --------------------------
left_idx = np.where(side_vals > 0)[0]   # hanging
right_idx = np.where(side_vals < 0)[0]  # footwall

side_points = {
    'hanging': { 'xy': XY[left_idx], 'idx': left_idx },
    'footwall': { 'xy': XY[right_idx], 'idx': right_idx }
}

tri_side = {}
for side_name, info in side_points.items():
    pts2d = info['xy']
    if pts2d.shape[0] < 3:
        tri = None
    else:
        tri = Delaunay(pts2d)
    tri_side[side_name] = tri

# -------------------------- 构建断层 ribbon（三角带） --------------------------
def build_fault_ribbon(polyline, throws_at_vertices, offset_width=3.0):
    """
    通过在中轴线两侧沿法线偏移构造窄带，然后逐段生成四边->两三角的网格。
    返回 ribbon_xy (M,2), ribbon_z (M,), triangles (K,3) (索引到 ribbon_xy)
    """
    n = polyline.shape[0]
    left_pts = []
    right_pts = []
    for i in range(n):
        if i == 0:
            tvec = polyline[1] - polyline[0]
        elif i == n-1:
            tvec = polyline[-1] - polyline[-2]
        else:
            tvec = polyline[i+1] - polyline[i-1]
        tvec = tvec / np.linalg.norm(tvec)
        normal = np.array([-tvec[1], tvec[0]])
        left = polyline[i] + normal * offset_width
        right = polyline[i] - normal * offset_width
        left_pts.append(left)
        right_pts.append(right)

    ribbon_xy = np.vstack([np.array(left_pts), np.array(right_pts[::-1])])
    m = n
    triangles = []
    for i in range(m-1):
        a = i
        b = i+1
        c = 2*m - i - 1
        d = 2*m - i - 2
        triangles.append([a, b, c])
        triangles.append([a, c, d])
    triangles = np.array(triangles, dtype=int)

    # Z 值：左侧（对应中轴 vertex）使用 throws_at_vertices 加基准高度，右侧使用基准高度
    basez = np.mean([layers["Bedrock"].mean(), layers["Coal"].mean(), layers["Topsoil"].mean()])
    left_z = basez + throws_at_vertices
    right_z = np.full_like(left_z, basez)
    ribbon_z = np.concatenate([left_z, right_z[::-1]])
    return ribbon_xy, ribbon_z, triangles

ribbon_xy, ribbon_z, ribbon_tris = build_fault_ribbon(fault_xy, throws_at_vertices, offset_width=6.0)

# -------------------------- 保存示例数据 CSV --------------------------
# 前 12 个点作为示例表
sample_n = min(12, N)
df_sample = pd.DataFrame({
    "X": XY[:sample_n,0],
    "Y": XY[:sample_n,1],
    "Bedrock_Z": layers["Bedrock"][:sample_n],
    "Coal_Z": layers["Coal"][:sample_n],
    "Topsoil_Z": layers["Topsoil"][:sample_n],
    "t_along_fault": t_vals[:sample_n],
    "side_sign": side_vals[:sample_n],
    "distance_to_fault": dist_vals[:sample_n]
})
df_sample.to_csv(os.path.join(OUTDIR, "points_sample.csv"), index=False)

# footwall 三角索引（若存在）
if tri_side['footwall'] is not None:
    df_tri = pd.DataFrame(tri_side['footwall'].simplices[:200], columns=['i','j','k'])
    df_tri.to_csv(os.path.join(OUTDIR, "footwall_tri_sample.csv"), index=False)

# 保存每层的偏移后点集（包含原索引）
for lname, Zarr in layers_offset.items():
    df = pd.DataFrame({
        "X": XY[:,0],
        "Y": XY[:,1],
        "Z": Zarr
    })
    df.to_csv(os.path.join(OUTDIR, f"{lname}_subsided.csv"), index=False)

# -------------------------- 3D 可视化并保存 PNG --------------------------
layer_to_plot = "Topsoil"  # 只绘制这一层

fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"TIN（单层: {layer_to_plot}）含 Z 字形断层", fontsize=14)

# 绘制选定层的悬挂盘与足盘
Zarr = layers_offset[layer_to_plot]
for side_name, info in side_points.items():
    idxs = info['idx']
    xy_side = info['xy']
    if tri_side[side_name] is None:
        continue
    simplices = tri_side[side_name].simplices
    Z_side = Zarr[idxs]
    tri = Triangulation(xy_side[:,0], xy_side[:,1], simplices)
    ax.plot_trisurf(tri, Z_side, linewidth=0.2, antialiased=True, alpha=0.85)

# 绘制断层 ribbon
tri_r = Triangulation(ribbon_xy[:,0], ribbon_xy[:,1], ribbon_tris)
ax.plot_trisurf(tri_r, ribbon_z, linewidth=0.5, antialiased=True, color='lightgray', alpha=1.0)

# 绘制断层中轴线
ax.plot(fault_xy[:,0], fault_xy[:,1], 
        np.full(fault_xy.shape[0], np.mean(Zarr)),
        linewidth=3.0, color='k', label='fault centerline')

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.view_init(elev=30, azim=-60)
plt.tight_layout()

png_path = os.path.join(OUTDIR, f"tin_fault_visualization_{layer_to_plot}.png")
plt.savefig(png_path, dpi=300)
plt.show()
print(f"已生成单层可视化文件: {png_path}")
