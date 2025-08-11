# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import math
import gempy as gp
import matplotlib.pyplot as plt

# -----------------------
# 辅助：从剖面点（2D）通过三次样条拟合并计算切向量，进而得到方位(azimuth,dip)
# 论文中做法：用 WebPlotDigitizer 得到二维剖面上岩性段的离散点，
# 用三次样条拟合并取一阶导数作为切向量，再转为法向量与方位角/倾角。（见论文方法）
# 引自论文说明。:contentReference[oaicite:5]{index=5}
# -----------------------
def profile_points_to_orientations(profile_xyz, global_profile_azimuth=None, sample_step=5):
    """
    profile_xyz: Nx3 array/list of points along the profile (in map coords x,y and elevation z)
                 The order should follow the profile line.
    global_profile_azimuth: if the profile line has an overall azimuth in map plane, you can pass it;
                            otherwise we compute local tangents and convert.
    Returns: DataFrame with columns x,y,z, azimuth(deg), dip(deg), polarity(+1)
    """
    profile_xyz = np.asarray(profile_xyz)
    # param t as cumulative chord length along profile
    diffs = np.diff(profile_xyz[:, :2], axis=0)
    seglen = np.sqrt((diffs**2).sum(axis=1))
    t = np.concatenate(([0.0], np.cumsum(seglen)))
    # fit cubic splines x(t), y(t), z(t) separately
    cs_x = CubicSpline(t, profile_xyz[:,0])
    cs_y = CubicSpline(t, profile_xyz[:,1])
    cs_z = CubicSpline(t, profile_xyz[:,2])

    # sample along profile at regular t increments (adjust sample_step)
    t_sample = np.linspace(t[0], t[-1], max(3, int((t[-1]-t[0])/sample_step)))
    xs = cs_x(t_sample)
    ys = cs_y(t_sample)
    zs = cs_z(t_sample)

    # first derivatives (tangent vector in 3D)
    dx = cs_x.derivative()(t_sample)
    dy = cs_y.derivative()(t_sample)
    dz = cs_z.derivative()(t_sample)

    records = []
    for xi, yi, zi, dxi, dyi, dzi in zip(xs, ys, zs, dx, dy, dz):
        # horizontal azimuth (bearing) of tangent vector measured clockwise from north (Y axis)
        # careful: paper uses Y-axis as reference; adapt if your coords differ.
        # We'll compute azimuth measured in degrees: 0 = north (positive Y), increase clockwise.
        # compute atan2 of (dx, dy) but adjust to 0..360 with Y as 0:
        # normally atan2(dx, dy) gives angle from Y axis.
        az_rad = math.atan2(dxi, dyi)  # dx,dy swapped so 0 points to positive Y
        az_deg = (math.degrees(az_rad) + 360) % 360

        # dip: angle between tangent vector and horizontal plane
        horiz_len = math.hypot(dxi, dyi)
        dip_rad = math.atan2(dzi, horiz_len)
        dip_deg = math.degrees(abs(dip_rad))  # dip is positive magnitude

        # polarity: sign indicating which side is 'younger' in GemPy convention.
        # Paper uses +1 / -1; here we default to +1. You may need to decide per formation.
        polarity = 1

        records.append((xi, yi, zi, az_deg, dip_deg, polarity))

    df = pd.DataFrame(records, columns=['X','Y','Z','azimuth','dip','polarity'])
    return df

# -----------------------
# 若你已经有 orientations.csv、interfaces.csv（论文表 1/2），可以直接跳到 GemPy 部分。
# -----------------------

# 示例：如果你只有剖面点（从 WebPlotDigitizer 导出），可以走这个流程：
# profile_points = np.loadtxt('profile_points_from_webplot.txt')  # (N,3)
# orientations_df = profile_points_to_orientations(profile_points)
# orientations_df.to_csv('orientations_generated_from_profile.csv', index=False)

# -----------------------
# GemPy 建模代码（模板）
# 依据 GemPy 官方教程，使用 gp.create_model / gp.init_data / gp.add_surface_points / gp.add_orientations / gp.compute_model
# 论文中也是用这一类流程来创建模型并可视化（paper 方法节）。:contentReference[oaicite:6]{index=6}
# -----------------------

def run_gempy_model(interfaces_csv='interfaces.csv', orientations_csv='orientations.csv',
                    model_name='Yantan_model', extent=None, resolution=[80,80,80]):
    """
    interfaces_csv: path to CSV with columns x,y,z,formation (表1)
    orientations_csv: path to CSV with columns x,y,z,azimuth,dip,polarity,formation (表2)
    extent: [xmin, xmax, ymin, ymax, zmin, zmax] ; if None, computed from data with small padding
    resolution: grid resolution per axis (<=100 each per paper limitation)
    """
    # load data
    interfaces = pd.read_csv(interfaces_csv)
    orientations = pd.read_csv(orientations_csv)

    # compute extent if not provided
    if extent is None:
        pad = 10
        xmin, xmax = interfaces['X'].min()-pad, interfaces['X'].max()+pad
        ymin, ymax = interfaces['Y'].min()-pad, interfaces['Y'].max()+pad
        zmin, zmax = interfaces['Z'].min()-pad, interfaces['Z'].max()+pad
        extent = [xmin, xmax, ymin, ymax, zmin, zmax]

    # create model (GemPy >=3 uses create_geomodel)
    geo_model = gp.create_model(model_name)

    # init data into model (surface points and orientations DataFrame)
    # 用法参考 GemPy 文档：gp.init_data(geo_model, extent, resolution, surface_points_df=..., orientations_df=...)
    gp.init_data(geo_model, extent, resolution,
                 surface_points_df=interfaces,
                 orientations_df=orientations,
                 default_values=True)
    # 可选：查看已载入的界面和方位
    print("Surfaces (interfaces) loaded:\n", geo_model.surfaces.head())
    print("Orientations loaded:\n", geo_model.orientations.head())

    # 设定堆叠/层序（若需要），示例：gp.map_stack_to_surfaces 或 gp.set_series_order ...
    # 论文中通过 gp.map_stack_to_surfaces / gp.map_stack_to_series 或类似函数设置序列关系。
    # 这里给出一个常见流程：先定义 series 及颜色（可按论文表 3 调整）
    # gp.map_stack_to_surfaces(geo_model, stack_dict)  # 如果需要

    # 设置插值参数（可选：调整 variogram / kriging 参数 等）
    # 论文给出了 kriging range, nugget 等参数（示例见表 4），这些可以在 gp.set_interpolation_data 中指定。
    # 例如： gp.set_interpolation_data(geo_model, compile_theano=True, ...)

    # 运行模型计算（插值 + 等值面提取）
    gp.set_interpolation_data(geo_model, compile_theano=True)
    gp.compute_model(geo_model)

    p3d = gp.plot_3d(geo_model)   # 论文使用 PyVista 内置 viewer 来查看三维模型。:contentReference[oaicite:7]{index=7}
    p2d = gp.plot_2d(geo_model, show=False)
    plt.show()

    return geo_model

# -----------------------
# 如果你要直接运行（假设已有 csv）
# -----------------------
if __name__ == '__main__':
    # 直接运行会尝试加载两个 csv; 若你只有剖面点，请先运行 profile -> orientations
    # 示例（占位）:
    run_gempy_model('interfaces.csv', 'orientations.csv', resolution=[80,80,80])

    print("请编辑脚本顶部的 CSV 路径并运行。脚本包含：\n - 从剖面点生成方位的函数\n - GemPy 建模流程模板\n参见论文中关于数据表结构（表1/表2）与 kriging 参数（表4）的描述以设置参数。:contentReference[oaicite:8]{index=8}")
