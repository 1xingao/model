# -*- coding: utf-8 -*-
from collections import defaultdict
import pandas as pd

# 示例钻孔数据（每个钻孔内的地层从上至下）
boreholes = {
    # "B1": ["黄土","细粒砂岩","泥岩","中粒砂岩","砂质泥岩","泥岩","细粒砂岩"],
    "B2": ["黄土","砂质泥岩"],
    "B3": ["黄土","砂质泥岩","细粒砂岩","砂质泥岩","泥岩","细粒砂岩"],
    "B4": ["黄土","砾石","砂质泥岩","中粒砂岩"],
    # "B5": ["黄土","泥岩","细粒砂岩","泥岩","细粒砂岩","泥岩"]
}

# 1. 提取所有地层类型
all_layers = set(layer for layers in boreholes.values() for layer in layers)
all_layers = sorted(list(all_layers))  # 保持可读性排序
layer_index = {layer: i for i, layer in enumerate(all_layers)}

# 2. 初始化关系矩阵：delta[i][j] 累加
C = [[0 for _ in all_layers] for _ in all_layers]

# 3. 统计所有钻孔中地层的相对关系
for bh_layers in boreholes.values():
    for i in range(len(bh_layers)):
        for j in range(len(bh_layers)):
            if i == j:
                continue
            li, lj = bh_layers[i], bh_layers[j]
            idx_i, idx_j = layer_index[li], layer_index[lj]
            # 如果 i 在 j 上
            if i < j:
                C[idx_i][idx_j] += 1
            else:
                C[idx_i][idx_j] -= 1

# 4. 计算每一层的总优先级（上方越多，得分越高）
priority = {layer: 0 for layer in all_layers}
for i, li in enumerate(all_layers):
    for j, lj in enumerate(all_layers):
        priority[li] += C[i][j]

# 5. 排序得出统一层序（得分从高到低）
sorted_layers = sorted(priority.items(), key=lambda x: -x[1])
unified_sequence = [layer for layer, _ in sorted_layers]

print("统一地层层序：")
print(unified_sequence)

# 6. 补全所有钻孔的层序，插入虚拟地层（厚度为 0）
standardized_boreholes = {}

for bh_name, bh_layers in boreholes.items():
    completed = []
    for layer in unified_sequence:
        if layer in bh_layers:
            completed.append(layer)
        else:
            completed.append(f"{layer}_虚拟")  # 用后缀标识虚拟层
    standardized_boreholes[bh_name] = completed

print("\n标准化后的钻孔层序（含虚拟层）：")
df = pd.DataFrame(standardized_boreholes)
print(df)
