# 差分进化算法优化器常量定义
DE_GENERATIONS = 500
DE_POPULATION_SIZE = 50
DE_CROSSOVER_RATE = 0.7
DE_DIFFERENTIAL_WEIGHT = 0.8
DE_STRATEGY = 'rand/1/bin'

import numpy as np
from base_optimizer import BaseKrigeOptimizer


class DEKrigeOptimizer(BaseKrigeOptimizer):
    """
    基于差分进化算法的克里金插值参数优化器
    """
    
    def __init__(self, generations=DE_GENERATIONS,
                 population_size=DE_POPULATION_SIZE,
                 crossover_rate=DE_CROSSOVER_RATE,
                 differential_weight=DE_DIFFERENTIAL_WEIGHT,
                 strategy=DE_STRATEGY):
        """
        初始化差分进化算法优化器
        
        参数:
            generations: 迭代代数
            population_size: 种群大小
            crossover_rate: 交叉概率
            differential_weight: 差分权重
            strategy: 变异策略
        """
        super().__init__()
        self.generations = generations
        self.population_size = population_size
        self.CR = crossover_rate
        self.F = differential_weight
        self.strategy = strategy
        
    def initialize_population(self, bounds):
        """
        初始化种群
        
        参数:
            bounds: 参数边界字典
            
        返回:
            population: 初始种群矩阵 (population_size, 3)
        """
        population = np.zeros((self.population_size, 3))
        
        param_names = ['nugget', 'range', 'sill']
        for i, param in enumerate(param_names):
            low, high = bounds[param]
            population[:, i] = np.random.uniform(low, high, self.population_size)
        
        return population
    
    def mutation(self, population, target_idx, bounds):
        """
        变异操作
        
        参数:
            population: 当前种群
            target_idx: 目标个体索引
            bounds: 参数边界
            
        返回:
            mutant: 变异向量
        """
        pop_size = len(population)
        
        # 随机选择三个不同的个体（不包括目标个体）
        candidates = list(range(pop_size))
        candidates.remove(target_idx)
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        
        # 差分变异: mutant = x_r1 + F * (x_r2 - x_r3)
        mutant = population[r1] + self.F * (population[r2] - population[r3])
        
        # 边界约束
        param_names = ['nugget', 'range', 'sill']
        for i, param in enumerate(param_names):
            low, high = bounds[param]
            mutant[i] = np.clip(mutant[i], low, high)
        
        return mutant
    
    def crossover(self, target, mutant):
        """
        交叉操作
        
        参数:
            target: 目标向量
            mutant: 变异向量
            
        返回:
            trial: 试验向量
        """
        trial = target.copy()
        
        # 确保至少有一个参数来自变异向量
        rand_j = np.random.randint(0, 3)
        
        for j in range(3):
            if np.random.rand() < self.CR or j == rand_j:
                trial[j] = mutant[j]
        
        return trial
    
    def selection(self, target, trial, train_data, test_data, objective):
        """
        选择操作
        
        参数:
            target: 目标个体
            trial: 试验个体
            train_data: 训练数据
            test_data: 测试数据
            objective: 目标函数类型
            
        返回:
            selected: 选择的个体
            score: 对应的适应度
        """
        x_train, y_train, z_train = train_data
        x_test, y_test, z_test = test_data
        
        # 计算目标个体适应度
        target_score = self.evaluate_fitness(
            target[0], target[1], target[2],
            x_train, y_train, z_train,
            x_test, y_test, z_test,
            objective
        )
        
        # 计算试验个体适应度
        trial_score = self.evaluate_fitness(
            trial[0], trial[1], trial[2],
            x_train, y_train, z_train,
            x_test, y_test, z_test,
            objective
        )
        
        # 选择更好的个体
        if trial_score <= target_score:
            return trial, trial_score
        else:
            return target, target_score
    
    def optimize(self, train_data, test_data, bounds, objective='rmse', verbose=True):
        """
        执行差分进化算法优化
        
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
        # 初始化种群
        population = self.initialize_population(bounds)
        
        # 记录最优解历史
        best_scores_history = []
        
        for generation in range(self.generations):
            # 新一代种群
            new_population = []
            generation_scores = []
            
            for i in range(self.population_size):
                # 变异
                mutant = self.mutation(population, i, bounds)
                
                # 交叉
                trial = self.crossover(population[i], mutant)
                
                # 选择
                selected, score = self.selection(
                    population[i], trial, train_data, test_data, objective
                )
                
                new_population.append(selected)
                generation_scores.append(score)
                
                # 更新全局最优
                if score < self.best_score:
                    self.best_score = score
                    self.best_nugget = selected[0]
                    self.best_range = selected[1]
                    self.best_sill = selected[2]
            
            population = np.array(new_population)
            best_scores_history.append(self.best_score)
            
            # 打印进度
            if verbose and generation % 50 == 0:
                avg_score = np.mean(generation_scores)
                print(f"代数 {generation}: 最优适应度 = {self.best_score:.6f}, "
                      f"平均适应度 = {avg_score:.6f}")
        
        if verbose:
            print(f"差分进化优化完成，最优适应度: {self.best_score:.6f}")
        
        return self.best_score, self.get_best_parameters()

# 差分进化算法基本逻辑说明：
# 1. 初始化：随机生成初始种群，每个个体包含nugget、range、sill三个参数
# 2. 变异：对每个目标个体，随机选择三个不同个体进行差分变异操作
# 3. 交叉：将变异向量与目标向量进行交叉，产生试验向量
# 4. 选择：比较目标个体和试验个体的适应度，选择更好的进入下一代
# 5. 迭代：重复2-4步直到达到最大代数
# 6. 优点：参数少、收敛快、对初值不敏感、适合连续优化
# 7. 特点：基于种群差异信息进行搜索，自适应性强
# 8. 适用场景：连续参数优化，特别是多模态和高维优化问题
