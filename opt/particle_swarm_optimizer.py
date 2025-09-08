# 粒子群算法优化器常量定义
PSO_ITERATIONS = 500
PSO_PARTICLES = 30
PSO_INERTIA_WEIGHT = 0.7
PSO_COGNITIVE_COEFF = 1.5
PSO_SOCIAL_COEFF = 1.5
PSO_VELOCITY_LIMIT = 0.2

import numpy as np
from base_optimizer import BaseKrigeOptimizer


class PSOKrigeOptimizer(BaseKrigeOptimizer):
    """
    基于粒子群算法的克里金插值参数优化器
    """
    
    def __init__(self, iterations=PSO_ITERATIONS,
                 n_particles=PSO_PARTICLES,
                 inertia_weight=PSO_INERTIA_WEIGHT,
                 cognitive_coeff=PSO_COGNITIVE_COEFF,
                 social_coeff=PSO_SOCIAL_COEFF,
                 velocity_limit=PSO_VELOCITY_LIMIT):
        """
        初始化粒子群算法优化器
        
        参数:
            iterations: 迭代次数
            n_particles: 粒子数量
            inertia_weight: 惯性权重
            cognitive_coeff: 认知系数（个体最优影响）
            social_coeff: 社会系数（全局最优影响）
            velocity_limit: 速度限制比例
        """
        super().__init__()
        self.iterations = iterations
        self.n_particles = n_particles
        self.w = inertia_weight
        self.c1 = cognitive_coeff
        self.c2 = social_coeff
        self.velocity_limit = velocity_limit
        
    def initialize_swarm(self, bounds):
        """
        初始化粒子群
        
        参数:
            bounds: 参数边界字典
            
        返回:
            positions: 粒子位置矩阵 (n_particles, 3)
            velocities: 粒子速度矩阵 (n_particles, 3)
        """
        positions = np.zeros((self.n_particles, 3))
        velocities = np.zeros((self.n_particles, 3))
        
        # 初始化位置
        param_names = ['nugget', 'range', 'sill']
        for i, param in enumerate(param_names):
            low, high = bounds[param]
            positions[:, i] = np.random.uniform(low, high, self.n_particles)
            
            # 初始化速度（范围的一定比例）
            velocity_range = (high - low) * self.velocity_limit
            velocities[:, i] = np.random.uniform(
                -velocity_range, velocity_range, self.n_particles
            )
        
        return positions, velocities
    
    def update_velocity(self, velocities, positions, personal_best, global_best, bounds):
        """
        更新粒子速度
        
        参数:
            velocities: 当前速度
            positions: 当前位置
            personal_best: 个体最优位置
            global_best: 全局最优位置
            bounds: 参数边界
            
        返回:
            updated_velocities: 更新后的速度
        """
        r1 = np.random.rand(self.n_particles, 3)
        r2 = np.random.rand(self.n_particles, 3)
        
        # PSO速度更新公式
        new_velocities = (
            self.w * velocities +
            self.c1 * r1 * (personal_best - positions) +
            self.c2 * r2 * (global_best - positions)
        )
        
        # 速度限制
        param_names = ['nugget', 'range', 'sill']
        for i, param in enumerate(param_names):
            low, high = bounds[param]
            velocity_limit = (high - low) * self.velocity_limit
            new_velocities[:, i] = np.clip(
                new_velocities[:, i], -velocity_limit, velocity_limit
            )
        
        return new_velocities
    
    def update_position(self, positions, velocities, bounds):
        """
        更新粒子位置
        
        参数:
            positions: 当前位置
            velocities: 当前速度
            bounds: 参数边界
            
        返回:
            updated_positions: 更新后的位置
        """
        new_positions = positions + velocities
        
        # 边界约束
        param_names = ['nugget', 'range', 'sill']
        for i, param in enumerate(param_names):
            low, high = bounds[param]
            new_positions[:, i] = np.clip(new_positions[:, i], low, high)
        
        return new_positions
    
    def optimize(self, train_data, test_data, bounds, objective='rmse', verbose=True):
        """
        执行粒子群算法优化
        
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
        
        # 初始化粒子群
        positions, velocities = self.initialize_swarm(bounds)
        
        # 初始化个体最优和全局最优
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.n_particles, float('inf'))
        
        global_best_position = None
        global_best_score = float('inf')
        
        # 记录最优解历史
        best_scores_history = []
        
        for iteration in range(self.iterations):
            # 计算适应度
            for i in range(self.n_particles):
                score = self.evaluate_fitness(
                    positions[i, 0], positions[i, 1], positions[i, 2],
                    x_train, y_train, z_train,
                    x_test, y_test, z_test,
                    objective
                )
                
                # 更新个体最优
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                
                # 更新全局最优
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i].copy()
                    
                    # 更新类属性
                    self.best_score = global_best_score
                    self.best_nugget = global_best_position[0]
                    self.best_range = global_best_position[1]
                    self.best_sill = global_best_position[2]
            
            best_scores_history.append(global_best_score)
            
            # 打印进度
            if verbose and iteration % 50 == 0:
                print(f"迭代 {iteration}: 最优适应度 = {global_best_score:.6f}")
            
            # 更新速度和位置
            velocities = self.update_velocity(
                velocities, positions, personal_best_positions, 
                global_best_position, bounds
            )
            positions = self.update_position(positions, velocities, bounds)
        
        if verbose:
            print(f"粒子群优化完成，最优适应度: {global_best_score:.6f}")
        
        return global_best_score, self.get_best_parameters()

# 粒子群算法基本逻辑说明：
# 1. 初始化：随机生成粒子群的初始位置和速度，每个粒子代表一组参数
# 2. 评估：计算每个粒子的适应度（预测误差）
# 3. 更新记录：更新每个粒子的个体最优位置和全局最优位置
# 4. 速度更新：根据PSO公式更新粒子速度，考虑惯性、个体经验、群体经验
# 5. 位置更新：根据速度更新粒子位置，并进行边界约束
# 6. 迭代：重复2-5步直到达到最大迭代次数
# 7. 优点：收敛速度快，参数少，易于实现和调试
# 8. 适用场景：连续优化问题，需要快速找到较好解的情况
