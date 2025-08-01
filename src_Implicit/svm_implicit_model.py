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

data_path = os.path.join(os.path.dirname(__file__), '../data/随机分布钻孔数据.xlsx')
df = pd.read_excel(data_path)


points = []
labels = []
z_top = 0  # 初始地层顶面高度
for _, row in df.iterrows():

    x, y = row['X'], row['Y']
    if row['地层'] == '填土':
        z_top = 0
    thickness = row['厚度']*10
    z_bottom = z_top + thickness

    for frac in np.linspace(0, 1, 5):
        z = z_top + frac * (z_bottom - z_top)
        points.append([x, y, z])
        labels.append(row['地层'])
    z_top = z_bottom  # 更新地层顶面高度
points = np.array(points)
labels = np.array(labels)


le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

x_min, x_max = points[:,0].min(), points[:,0].max()
y_min, y_max = points[:,1].min(), points[:,1].max()
z_min, z_max = points[:,2].min(), points[:,2].max()
grid_res = 100 # 可调
xx, yy, zz = np.meshgrid(
    np.linspace(x_min, x_max, grid_res),
    np.linspace(y_min, y_max, grid_res),
    np.linspace(z_min, z_max, grid_res),
    indexing='ij'   # 关键：保证三维顺序与pyvista一致
)
grid_points = np.c_[xx.ravel(order='C'), yy.ravel(order='C'), zz.ravel(order='C')]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(points)
grid_scaled = scaler.transform(grid_points)
svm = SVC(kernel='rbf', gamma='scale')
svm.fit(X_scaled, labels_encoded)


pred_labels = svm.predict(grid_scaled)
pred_labels = pred_labels.reshape(xx.shape, order='C')


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


grid_points_df = pd.DataFrame(grid_points, columns=['X', 'Y', 'Z'])
grid_points_df['Labels'] = pred_labels.ravel(order='C')
point_cloud = pv.PolyData(grid_points_df[['X', 'Y', 'Z']].values)


csv_path = os.path.join(os.path.dirname(__file__), '../data/point_cloud.csv')
grid_points_df.to_csv(csv_path, index=False)
print(f"点云数据已保存为 CSV 格式: {csv_path}")


