from sko.ACA import ACA_TSP
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt

num_points = 300

# 用随机数生成 num_points 个点
points_coordinate = np.random.rand(num_points, 2)
# 调用 scipy 自动计算点与点之间的欧拉距离，生成距离矩阵
distance_matrix = spatial.distance.cdist(
    points_coordinate, points_coordinate, metric='euclidean')

def cal_total_distance(routine):
    num_points, = routine.shape
    # 计算距离和
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=450, max_iter=200,
              distance_matrix=distance_matrix
              )

best_x, best_y = aca.run()
print('best_x: \n', best_x, '\n', 'best_y: ', best_y)

fig, ax = plt.subplots(1, 2)
best_circuit = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_circuit, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r', markersize=3)
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
plt.show()