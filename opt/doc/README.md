# 克里金插值参数优化算法库

本库实现了多种优化算法来自动优化克里金插值的变差函数参数，提高插值精度。

## 算法简介

### 1. 遗传算法 (GA)
- **文件**: `genetic_algorithm_optimizer.py`
- **原理**: 模拟生物进化过程，通过选择、交叉、变异操作寻找最优参数
- **优点**: 全局搜索能力强，适合处理非凸优化问题
- **适用场景**: 参数空间复杂，存在多个局部最优解

### 2. 粒子群算法 (PSO)
- **文件**: `particle_swarm_optimizer.py`
- **原理**: 模拟鸟群觅食行为，粒子在解空间中根据个体和群体经验移动
- **优点**: 收敛速度快，参数少，易于实现
- **适用场景**: 连续优化问题，需要快速找到较好解

### 3. 蚁群算法 (ACO)
- **文件**: `ant_colony_optimizer.py`
- **原理**: 模拟蚂蚁觅食过程，通过信息素机制指导搜索方向
- **优点**: 具有正反馈机制，能找到多个较好解
- **适用场景**: 离散参数空间，需要探索多个可行解

### 4. 差分进化算法 (DE)
- **文件**: `differential_evolution_optimizer.py`
- **原理**: 基于种群差异信息进行搜索，通过变异、交叉、选择操作进化
- **优点**: 参数少、收敛快、对初值不敏感
- **适用场景**: 连续参数优化，特别是多模态和高维问题

### 5. 贝叶斯优化 (BO)
- **文件**: `bayesian_optimizer.py`
- **原理**: 使用高斯过程建模目标函数，通过采集函数平衡探索和利用
- **优点**: 样本效率高，适合昂贵目标函数优化
- **适用场景**: 目标函数评估代价高，需要少量样本找到最优解

## 文件结构

```
opt/
├── base_optimizer.py              # 基类，提供公共功能
├── genetic_algorithm_optimizer.py # 遗传算法
├── particle_swarm_optimizer.py   # 粒子群算法
├── ant_colony_optimizer.py       # 蚁群算法
├── differential_evolution_optimizer.py # 差分进化算法
├── bayesian_optimizer.py         # 贝叶斯优化
├── optimization_comparison.py    # 算法比较和可视化
├── test_algorithms.py           # 算法测试脚本
├── target.md                    # 需求说明文档
└── README.md                    # 本文件
```

## 使用方法

### 1. 单个算法使用

```python
from genetic_algorithm_optimizer import GAKrigeOptimizer

# 创建优化器
optimizer = GAKrigeOptimizer(generations=500, population_size=50)

# 加载和分割数据
train_data, test_data = optimizer.load_and_split_data(
    'data.xlsx', '松散层', train_ratio=0.7, random_seed=42
)

# 定义参数边界
x_train, y_train, z_train = train_data
bounds = optimizer.define_parameter_bounds(x_train, y_train, z_train)

# 运行优化
best_score, best_params = optimizer.optimize(
    train_data, test_data, bounds, objective='rmse'
)

print(f"最优参数: {best_params}")
```

### 2. 算法比较

```python
from optimization_comparison import KrigeOptimizationComparison

# 创建比较器
comparator = KrigeOptimizationComparison('data.xlsx', '松散层')

# 运行算法比较
results = comparator.run_algorithm_comparison(
    algorithms=['GA', 'PSO', 'DE'],
    train_ratio=0.7,
    random_seed=42
)

# 生成比较图
comparator.plot_semivariogram_comparison()
comparator.plot_interpolation_comparison()
comparator.plot_results_summary()

# 导出结果
comparator.export_results('results.json')
```

### 3. 快速测试

```python
# 运行测试脚本验证所有算法
python test_algorithms.py
```

## 参数配置

### 通用参数
- `objective`: 目标函数类型 ('rmse', 'mae', 'likelihood')
- `train_ratio`: 训练集比例 (默认: 0.7)
- `random_seed`: 随机种子 (默认: 42)

### 算法特定参数

#### 遗传算法 (GA)
- `generations`: 迭代代数 (默认: 500)
- `population_size`: 种群大小 (默认: 50)
- `crossover_rate`: 交叉概率 (默认: 0.8)
- `mutation_rate`: 变异概率 (默认: 0.2)

#### 粒子群算法 (PSO)
- `iterations`: 迭代次数 (默认: 500)
- `n_particles`: 粒子数量 (默认: 30)
- `inertia_weight`: 惯性权重 (默认: 0.7)
- `cognitive_coeff`: 认知系数 (默认: 1.5)
- `social_coeff`: 社会系数 (默认: 1.5)

#### 蚁群算法 (ACO)
- `iterations`: 迭代次数 (默认: 500)
- `n_ants`: 蚂蚁数量 (默认: 100)
- `pheromone_decay`: 信息素衰减系数 (默认: 0.8)
- `alpha`: 信息素重要程度 (默认: 1.0)
- `beta`: 启发式信息重要程度 (默认: 2.0)

#### 差分进化算法 (DE)
- `generations`: 迭代代数 (默认: 500)
- `population_size`: 种群大小 (默认: 50)
- `crossover_rate`: 交叉概率 (默认: 0.7)
- `differential_weight`: 差分权重 (默认: 0.8)

#### 贝叶斯优化 (BO)
- `iterations`: 优化迭代次数 (默认: 100)
- `init_samples`: 初始采样点数量 (默认: 10)
- `acquisition`: 采集函数类型 ('ei', 'ucb', 'poi')

## 输出格式

所有算法返回统一的结果格式：

```json
{
  "nugget": 0.15,
  "range": 320.5,
  "sill": 1.2,
  "score": 0.45
}
```

## 依赖库

- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- pykrige

## 安装依赖

```bash
pip install numpy pandas matplotlib scipy scikit-learn pykrige
```

## 注意事项

1. **数据格式**: 输入数据应为Excel文件，包含列：'地层名称', 'x', 'y', 'z'
2. **参数边界**: 算法会根据数据特征自动设置合理的参数搜索边界
3. **收敛性**: 不同算法有不同的收敛特性，建议根据具体问题选择合适的算法
4. **计算时间**: 贝叶斯优化样本效率最高，但单次评估时间较长；进化算法需要更多迭代但并行度高

## 算法选择建议

- **快速原型**: PSO或DE算法，收敛快，参数少
- **高精度要求**: GA算法，全局搜索能力强
- **样本效率**: 贝叶斯优化，适合昂贵的目标函数
- **离散问题**: ACO算法，适合组合优化
- **多目标**: 可扩展为多目标进化算法

## 扩展功能

1. **多目标优化**: 同时优化预测精度和不确定性
2. **约束优化**: 添加参数的物理约束条件
3. **并行计算**: 利用多核并行加速优化过程
4. **自适应参数**: 根据收敛情况动态调整算法参数

## 问题反馈

如遇到问题或需要新功能，请检查：
1. 数据格式是否正确
2. 参数设置是否合理
3. 依赖库是否正确安装
