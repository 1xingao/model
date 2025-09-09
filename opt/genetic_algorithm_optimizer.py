# 遗传算法优化器常量定义
GA_GENERATIONS = 500
GA_POPULATION_SIZE = 50
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.2
GA_ELITE_RATIO = 0.1

import numpy as np
from base_optimizer import BaseKrigeOptimizer


class GAKrigeOptimizer(BaseKrigeOptimizer):
    """
    基于遗传算法的克里金插值参数优化器
    """
    
    def __init__(self, generations=GA_GENERATIONS, 
                 population_size=GA_POPULATION_SIZE,
                 crossover_rate=GA_CROSSOVER_RATE, 
                 mutation_rate=GA_MUTATION_RATE,
                 elite_ratio=GA_ELITE_RATIO):
        """
        初始化遗传算法优化器
        
        参数:
            generations: 迭代代数
            population_size: 种群大小
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_ratio: 精英保留比例
        """
        super().__init__()
        self.generations = generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = int(population_size * elite_ratio)
        
    def initialize_population(self, bounds):
        """
        初始化种群
        
        参数:
            bounds: 参数边界字典
            
        返回:
            population: 初始种群矩阵 (population_size, 3)
        """
        population = np.zeros((self.population_size, 3))
        
        # nugget
        population[:, 0] = np.random.uniform(
            bounds['nugget'][0], bounds['nugget'][1], self.population_size
        )
        
        # range
        population[:, 1] = np.random.uniform(
            bounds['range'][0], bounds['range'][1], self.population_size
        )
        
        # sill
        population[:, 2] = np.random.uniform(
            bounds['sill'][0], bounds['sill'][1], self.population_size
        )
        
        return population
    
    def selection(self, population, fitness):
        """
        选择操作：锦标赛选择
        
        参数:
            population: 当前种群
            fitness: 适应度数组
            
        返回:
            selected: 选择的个体
        """
        selected_indices = []
        
        # 精英保留
        elite_indices = np.argsort(fitness)[:self.elite_size]
        selected_indices.extend(elite_indices)
        
        # 锦标赛选择填充剩余位置
        tournament_size = 3
        while len(selected_indices) < self.population_size:
            tournament_indices = np.random.choice(
                len(population), tournament_size, replace=False
            )
            winner_idx = tournament_indices[np.argmin(fitness[tournament_indices])]
            selected_indices.append(winner_idx)
        
        return population[selected_indices[:self.population_size]]
    
    def crossover(self, parent1, parent2):
        """
        交叉操作：算术交叉
        
        参数:
            parent1, parent2: 父代个体
            
        返回:
            child1, child2: 子代个体
        """
        if np.random.rand() < self.crossover_rate:
            # 算术交叉
            alpha = np.random.rand()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutation(self, individual, bounds):
        """
        变异操作：高斯变异
        
        参数:
            individual: 个体
            bounds: 参数边界
            
        返回:
            mutated: 变异后的个体
        """
        mutated = individual.copy()
        
        for i in range(3):
            if np.random.rand() < self.mutation_rate:
                # 高斯变异
                param_range = bounds[list(bounds.keys())[i]]
                std = (param_range[1] - param_range[0]) * 0.1
                mutated[i] += np.random.normal(0, std)
                
                # 边界约束
                mutated[i] = np.clip(mutated[i], param_range[0], param_range[1])
        
        return mutated
    
    def optimize(self, train_data, test_data, bounds, objective='rmse', verbose=True):
        """
        执行遗传算法优化
        
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
        
        # 初始化种群
        population = self.initialize_population(bounds)
        
        # 记录最优解历史
        best_scores_history = []
        
        for generation in range(self.generations):
            # 计算适应度
            fitness = np.array([
                self.evaluate_fitness(
                    ind[0], ind[1], ind[2],
                    x_train, y_train, z_train,
                    x_test, y_test, z_test,
                    objective
                )
                for ind in population
            ])
            
            # 更新最优解
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_score:
                self.best_score = fitness[best_idx]
                self.best_nugget = population[best_idx, 0]
                self.best_range = population[best_idx, 1]
                self.best_sill = population[best_idx, 2]
            
            best_scores_history.append(self.best_score)
            
            # 打印进度
            if verbose and generation % 50 == 0:
                print(f"代数 {generation}: 最优适应度 = {self.best_score:.6f}")
            
            # 选择
            selected = self.selection(population, fitness)
            
            # 生成下一代
            next_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.population_size]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1, bounds)
                child2 = self.mutation(child2, bounds)
                
                next_population.extend([child1, child2])
            
            population = np.array(next_population[:self.population_size])
        
        if verbose:
            print(f"遗传算法优化完成，最优适应度: {self.best_score:.6f}")
            
            # 比较默认参数和优化参数的误差
            self.compare_all_parameters(train_data, test_data)
        
        return self.best_score, self.get_best_parameters()

# 遗传算法基本逻辑说明：
# 1. 初始化：随机生成初始种群，每个个体包含nugget、range、sill三个参数
# 2. 评估：使用交叉验证计算每个个体的适应度（预测误差）
# 3. 选择：采用精英保留+锦标赛选择，确保优秀个体传递到下一代
# 4. 交叉：使用算术交叉，在父代个体间线性组合产生子代
# 5. 变异：高斯变异增加种群多样性，防止早熟收敛
# 6. 迭代：重复2-5步直到达到最大代数
# 7. 优点：全局搜索能力强，适合处理非凸优化问题
# 8. 适用场景：参数空间复杂，存在多个局部最优解的情况
