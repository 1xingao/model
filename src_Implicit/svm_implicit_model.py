import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pyvista as pv
from scipy.stats import binned_statistic_dd

plt.rcParams['font.sans-serif'] = ['SimHei']
# 1. 读取钻孔数据
data_path = os.path.join(os.path.dirname(__file__), '../data/随机分布钻孔数据.xlsx')
df = pd.read_excel(data_path)

# 2. 生成三维点集（每层分解为多个点，z为地层底面）
points = []
labels = []
for _, row in df.iterrows():
    x, y = row['X'], row['Y']
    z_top = row['Z_top'] if 'Z_top' in row else 0  # 若无Z_top则假定为0
    thickness = row['厚度']*100
    z_bottom = z_top + thickness
    # 采样若干点（可调，默认每层采样5个点）
    for frac in np.linspace(0, 1, 5):
        z = z_top + frac * (z_bottom - z_top)
        points.append([x, y, z])
        labels.append(row['地层'])
points = np.array(points)
labels = np.array(labels)

# 3. 标签编码
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# 4. 构建空间格网
x_min, x_max = points[:,0].min(), points[:,0].max()
y_min, y_max = points[:,1].min(), points[:,1].max()
z_min, z_max = points[:,2].min(), points[:,2].max()
grid_res = 50 # 可调
xx, yy, zz = np.meshgrid(
    np.linspace(x_min, x_max, grid_res),
    np.linspace(y_min, y_max, grid_res),
    np.linspace(z_min, z_max, grid_res),
    indexing='ij'   # 关键：保证三维顺序与pyvista一致
)
grid_points = np.c_[xx.ravel(order='C'), yy.ravel(order='C'), zz.ravel(order='C')]

# 5. SVM建模
scaler = StandardScaler()
X_scaled = scaler.fit_transform(points)
grid_scaled = scaler.transform(grid_points)
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_scaled, labels_encoded)

# 6. 预测格网属性
pred_labels = svm.predict(grid_scaled)
pred_labels = pred_labels.reshape(xx.shape, order='C')

# 7. 可视化（3D体素/等值面）
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
for i, name in enumerate(le.classes_):
    mask = pred_labels == i
    ax.scatter(xx[mask], yy[mask], zz[mask], s=1, label=name)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('SVM隐式三维地质建模')
ax.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.show()

# 8. PyVista三维可视化
grid_points_df = pd.DataFrame(grid_points, columns=['X', 'Y', 'Z'])
grid_points_df['Labels'] = pred_labels.ravel(order='C')
point_cloud = pv.PolyData(grid_points_df[['X', 'Y', 'Z']].values)

# 保存点云数据为 CSV 格式
csv_path = os.path.join(os.path.dirname(__file__), '../data/point_cloud.csv')
grid_points_df.to_csv(csv_path, index=False)
print(f"点云数据已保存为 CSV 格式: {csv_path}")

# 保存点云数据为 PLY 格式
ply_path = os.path.join(os.path.dirname(__file__), '../data/point_cloud.ply')
point_cloud.save(ply_path)
print(f"点云数据已保存为 PLY 格式: {ply_path}")

