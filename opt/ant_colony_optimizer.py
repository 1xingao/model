# 蚁群算法优化器常量定义
ACO_ITERATIONS = 500
ACO_ANTS = 100
ACO_PHEROMONE_DECAY = 0.8
ACO_ALPHA = 1.0  # 信息素重要程度
ACO_BETA = 2.0   # 启发式信息重要程度
ACO_GRID_SIZE = 10  # 每个参数的离散化网格大小

import numpy as np
from base_optimizer import BaseKrigeOptimizer


class ACOKrigeOptimizer(BaseKrigeOptimizer):
    """
    基于蚁群算法的克里金插值参数优化器
    """
    
    def __init__(self, iterations=ACO_ITERATIONS,
                 n_ants=ACO_ANTS,
                 pheromone_decay=ACO_PHEROMONE_DECAY,
                 alpha=ACO_ALPHA,
                 beta=ACO_BETA,
                 grid_size=ACO_GRID_SIZE):
        """
        初始化蚁群算法优化器
        
        参数:
            iterations: 迭代次数
            n_ants: 蚂蚁数量
            pheromone_decay: 信息素衰减系数
            alpha: 信息素重要程度
            beta: 启发式信息重要程度
            grid_size: 参数离散化网格大小
        """
        super().__init__()
        self.iterations = iterations
        self.n_ants = n_ants
        self.pheromone_decay = pheromone_decay
        self.alpha = alpha
        self.beta = beta
        self.grid_size = grid_size
        
    def discretize_parameters(self, bounds):
        """
        将连续参数空间离散化
        
        参数:
            bounds: 参数边界字典
            
        返回:
            param_grids: 离散化后的参数网格
        """
        param_grids = {}
        
        for param_name, (low, high) in bounds.items():
            param_grids[param_name] = np.linspace(low, high, self.grid_size)
        
        return param_grids
    
    def initialize_pheromones_and_heuristics(self):
        """
        初始化信息素矩阵和启发式信息矩阵
        
        返回:
            pheromones: 信息素矩阵 (grid_size, grid_size, grid_size)
            heuristics: 启发式信息矩阵 (grid_size, grid_size, grid_size)
        """
        # 初始化信息素为均匀分布
        pheromones = np.ones((self.grid_size, self.grid_size, self.grid_size))
        
        # 启发式信息初始化为均匀分布
        heuristics = np.ones((self.grid_size, self.grid_size, self.grid_size))
        
        return pheromones, heuristics
    
    def calculate_selection_probabilities(self, pheromones, heuristics):
        """
        计算蚂蚁选择路径的概率
        
        参数:
            pheromones: 信息素矩阵
            heuristics: 启发式信息矩阵
            
        返回:
            probabilities: 选择概率矩阵
        """
        # 计算吸引力：信息素^alpha * 启发式信息^beta
        attractiveness = (pheromones ** self.alpha) * (heuristics ** self.beta)
        
        # 归一化为概率
        total_attractiveness = np.sum(attractiveness)
        if total_attractiveness == 0:
            probabilities = np.ones_like(attractiveness) / attractiveness.size
        else:
            probabilities = attractiveness / total_attractiveness
        
        return probabilities
    
    def ant_construct_solution(self, probabilities, param_grids):
        """
        单只蚂蚁构建解
        
        参数:
            probabilities: 选择概率矩阵
            param_grids: 参数网格
            
        返回:
            solution: 选择的参数组合 (nugget, range, sill)
            indices: 对应的网格索引 (i, j, k)
        """
        # 将3D概率矩阵展平为1D
        flat_probs = probabilities.flatten()
        
        # 根据概率选择
        chosen_idx = np.random.choice(len(flat_probs), p=flat_probs)
        
        # 将1D索引转换为3D索引
        i, j, k = np.unravel_index(chosen_idx, probabilities.shape)
        
        # 获取对应的参数值
        nugget = param_grids['nugget'][i]
        range_val = param_grids['range'][j]
        sill = param_grids['sill'][k]
        
        return (nugget, range_val, sill), (i, j, k)
    
    def update_pheromones(self, pheromones, solutions, scores):
        """
        更新信息素
        
        参数:
            pheromones: 当前信息素矩阵
            solutions: 蚂蚁找到的解的索引列表
            scores: 对应的适应度分数
            
        返回:
            updated_pheromones: 更新后的信息素矩阵
        """
        # 信息素衰减
        pheromones *= self.pheromone_decay
        
        # 添加新信息素
        for (i, j, k), score in zip(solutions, scores):
            if score > 0:  # 避免除零错误
                delta_pheromone = 1.0 / score
                pheromones[i, j, k] += delta_pheromone
        
        return pheromones
    
    def optimize(self, train_data, test_data, bounds, objective='rmse', verbose=True):
        """
        执行蚁群算法优化
        
        参数:
            train_data: 训练数据 (x_train, y_train, z_train)
            test_data: 测试数据 (x_test, y_test, z_test)
            bounds: 参数边界
            objective: 目标函数类型
            verbose: 是否打印过程信息
            
        返回:
            best_score: 最优适应度
            best_params: 最优参数字典
        """
        x_train, y_train, z_train = train_data
        x_test, y_test, z_test = test_data
        
        # 离散化参数空间
        param_grids = self.discretize_parameters(bounds)
        
        # 初始化信息素和启发式信息
        pheromones, heuristics = self.initialize_pheromones_and_heuristics()
        
        # 记录最优解历史
        best_scores_history = []
        
        for iteration in range(self.iterations):
            # 存储本轮蚂蚁的解和分数
            ant_solutions = []
            ant_scores = []
            
            # 计算选择概率
            probabilities = self.calculate_selection_probabilities(pheromones, heuristics)
            
            # 每只蚂蚁构建解
            for ant in range(self.n_ants):
                solution, indices = self.ant_construct_solution(probabilities, param_grids)
                
                # 评估解的质量
                score = self.evaluate_fitness(
                    solution[0], solution[1], solution[2],
                    x_train, y_train, z_train,
                    x_test, y_test, z_test,
                    objective
                )
                
                ant_solutions.append(indices)
                ant_scores.append(score)
                
                # 更新全局最优
                if score < self.best_score:
                    self.best_score = score
                    self.best_nugget = solution[0]
                    self.best_range = solution[1]
                    self.best_sill = solution[2]
            
            best_scores_history.append(self.best_score)
            
            # 打印进度
            if verbose and iteration % 50 == 0:
                print(f"迭代 {iteration}: 最优适应度 = {self.best_score:.6f}")
            
            # 更新信息素
            pheromones = self.update_pheromones(pheromones, ant_solutions, ant_scores)
        
        if verbose:
            print(f"蚁群优化完成，最优适应度: {self.best_score:.6f}")
            
            # 比较默认参数和优化参数的误差
            self.compare_all_parameters(train_data, test_data)
        
        return self.best_score, self.get_best_parameters()

# 蚁群算法基本逻辑说明：
# 1. 离散化：将连续的参数空间离散化为网格，每个网格点代表一个参数组合
# 2. 初始化：初始化信息素矩阵和启发式信息矩阵，所有路径初始吸引力相同
# 3. 构建解：每只蚂蚁根据信息素和启发式信息的组合概率选择参数组合
# 4. 评估：计算每个解的适应度（预测误差）
# 5. 更新信息素：好的解会在对应路径上留下更多信息素
# 6. 信息素衰减：模拟信息素随时间挥发，避免算法陷入局部最优
# 7. 迭代：重复3-6步直到达到最大迭代次数
# 8. 优点：具有正反馈机制，能够找到多个较好解，适合组合优化
# 9. 适用场景：离散参数空间，需要探索多个可行解的情况
