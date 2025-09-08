# 贝叶斯优化器常量定义
BO_ITERATIONS = 100
BO_INIT_SAMPLES = 10
BO_ACQUISITION = 'ei'  # expected improvement
BO_XI = 0.01  # exploration parameter
BO_KAPPA = 2.576  # exploration parameter for ucb

import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from base_optimizer import BaseKrigeOptimizer

# 过滤高斯过程收敛警告
warnings.filterwarnings('ignore', category=Warning, module='sklearn.gaussian_process')


class BayesianKrigeOptimizer(BaseKrigeOptimizer):
    """
    基于贝叶斯优化的克里金插值参数优化器
    """
    
    def __init__(self, iterations=BO_ITERATIONS,
                 init_samples=BO_INIT_SAMPLES,
                 acquisition=BO_ACQUISITION,
                 xi=BO_XI,
                 kappa=BO_KAPPA):
        """
        初始化贝叶斯优化器
        
        参数:
            iterations: 优化迭代次数
            init_samples: 初始采样点数量
            acquisition: 采集函数类型 ('ei', 'ucb', 'poi')
            xi: 期望改进的探索参数
            kappa: 置信上界的探索参数
        """
        super().__init__()
        self.iterations = iterations
        self.init_samples = init_samples
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa
        
        # 初始化高斯过程
        # 调整噪声水平范围，避免收敛警告
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-3, 1e1))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-4,  # 增加数值稳定性
            normalize_y=True,
            n_restarts_optimizer=5  # 减少重启次数提高效率
        )
        
    def normalize_parameters(self, params, bounds):
        """
        将参数归一化到[0,1]区间
        
        参数:
            params: 参数值 (nugget, range, sill)
            bounds: 参数边界
            
        返回:
            normalized: 归一化后的参数
        """
        normalized = np.zeros(3)
        param_names = ['nugget', 'range', 'sill']
        
        for i, param_name in enumerate(param_names):
            low, high = bounds[param_name]
            normalized[i] = (params[i] - low) / (high - low)
        
        return normalized
    
    def denormalize_parameters(self, normalized_params, bounds):
        """
        将归一化参数还原到原始尺度
        
        参数:
            normalized_params: 归一化参数
            bounds: 参数边界
            
        返回:
            params: 原始尺度参数
        """
        params = np.zeros(3)
        param_names = ['nugget', 'range', 'sill']
        
        for i, param_name in enumerate(param_names):
            low, high = bounds[param_name]
            params[i] = normalized_params[i] * (high - low) + low
        
        return params
    
    def latin_hypercube_sampling(self, bounds, n_samples):
        """
        拉丁超立方采样
        
        参数:
            bounds: 参数边界
            n_samples: 采样点数量
            
        返回:
            samples: 采样点矩阵 (n_samples, 3)
        """
        samples = np.zeros((n_samples, 3))
        param_names = ['nugget', 'range', 'sill']
        
        for i, param_name in enumerate(param_names):
            low, high = bounds[param_name]
            # 拉丁超立方采样
            intervals = np.linspace(0, 1, n_samples + 1)
            samples[:, i] = np.random.uniform(intervals[:-1], intervals[1:])
            samples[:, i] = samples[:, i] * (high - low) + low
        
        # 随机打乱每一列
        for i in range(3):
            np.random.shuffle(samples[:, i])
        
        return samples
    
    def expected_improvement(self, X, gp, y_best):
        """
        期望改进采集函数
        
        参数:
            X: 候选点
            gp: 高斯过程模型
            y_best: 当前最优值
            
        返回:
            ei: 期望改进值
        """
        mu, sigma = gp.predict(X.reshape(1, -1), return_std=True)
        sigma = sigma[0]
        
        if sigma == 0:
            return 0
        
        improvement = y_best - mu - self.xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei[0]
    
    def upper_confidence_bound(self, X, gp):
        """
        置信上界采集函数
        
        参数:
            X: 候选点
            gp: 高斯过程模型
            
        返回:
            ucb: 置信上界值
        """
        mu, sigma = gp.predict(X.reshape(1, -1), return_std=True)
        ucb = -(mu[0] - self.kappa * sigma[0])  # 负号因为我们要最小化
        
        return ucb
    
    def probability_of_improvement(self, X, gp, y_best):
        """
        改进概率采集函数
        
        参数:
            X: 候选点
            gp: 高斯过程模型
            y_best: 当前最优值
            
        返回:
            poi: 改进概率
        """
        mu, sigma = gp.predict(X.reshape(1, -1), return_std=True)
        sigma = sigma[0]
        
        if sigma == 0:
            return 0
        
        improvement = y_best - mu - self.xi
        Z = improvement / sigma
        poi = norm.cdf(Z)
        
        return poi[0]
    
    def acquisition_function(self, X, gp, y_best):
        """
        采集函数
        
        参数:
            X: 候选点
            gp: 高斯过程模型
            y_best: 当前最优值
            
        返回:
            acquisition_value: 采集函数值
        """
        if self.acquisition == 'ei':
            return -self.expected_improvement(X, gp, y_best)  # 负号因为优化器求最小值
        elif self.acquisition == 'ucb':
            return -self.upper_confidence_bound(X, gp)
        elif self.acquisition == 'poi':
            return -self.probability_of_improvement(X, gp, y_best)
        else:
            raise ValueError(f"未知的采集函数: {self.acquisition}")
    
    def optimize_acquisition(self, gp, y_best, bounds):
        """
        优化采集函数找到下一个采样点
        
        参数:
            gp: 高斯过程模型
            y_best: 当前最优值
            bounds: 参数边界
            
        返回:
            next_point: 下一个采样点
        """
        # 多次随机初始化，选择最好的结果
        best_acquisition = float('inf')
        best_point = None
        
        for _ in range(10):
            # 随机初始化
            x0 = np.random.uniform(0, 1, 3)
            
            # 优化采集函数
            result = minimize(
                lambda x: self.acquisition_function(x, gp, y_best),
                x0,
                bounds=[(0, 1)] * 3,
                method='L-BFGS-B'
            )
            
            if result.fun < best_acquisition:
                best_acquisition = result.fun
                best_point = result.x
        
        return self.denormalize_parameters(best_point, bounds)
    
    def optimize(self, train_data, test_data, bounds, objective='rmse', verbose=True):
        """
        执行贝叶斯优化
        
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
        
        # 初始采样
        initial_samples = self.latin_hypercube_sampling(bounds, self.init_samples)
        
        X_observed = []
        y_observed = []
        
        # 评估初始样本
        for sample in initial_samples:
            score = self.evaluate_fitness(
                sample[0], sample[1], sample[2],
                x_train, y_train, z_train,
                x_test, y_test, z_test,
                objective
            )
            
            X_observed.append(self.normalize_parameters(sample, bounds))
            y_observed.append(score)
            
            # 更新最优解
            if score < self.best_score:
                self.best_score = score
                self.best_nugget = sample[0]
                self.best_range = sample[1]
                self.best_sill = sample[2]
        
        X_observed = np.array(X_observed)
        y_observed = np.array(y_observed)
        
        # 贝叶斯优化迭代
        for iteration in range(self.iterations):
            try:
                # 拟合高斯过程
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.gp.fit(X_observed, y_observed)
                
                # 找到下一个采样点
                y_best = np.min(y_observed)
                next_point = self.optimize_acquisition(self.gp, y_best, bounds)
                
                # 评估新点
                score = self.evaluate_fitness(
                    next_point[0], next_point[1], next_point[2],
                    x_train, y_train, z_train,
                    x_test, y_test, z_test,
                    objective
                )
                
                # 更新观测数据
                next_point_normalized = self.normalize_parameters(next_point, bounds)
                X_observed = np.vstack([X_observed, next_point_normalized])
                y_observed = np.append(y_observed, score)
                
                # 更新最优解
                if score < self.best_score:
                    self.best_score = score
                    self.best_nugget = next_point[0]
                    self.best_range = next_point[1]
                    self.best_sill = next_point[2]
                
                # 打印进度
                if verbose and iteration % 10 == 0:
                    print(f"迭代 {iteration}: 最优适应度 = {self.best_score:.6f}")
                    
            except Exception as e:
                if verbose:
                    print(f"迭代 {iteration} 出现异常，跳过: {str(e)}")
                continue
        
        if verbose:
            print(f"贝叶斯优化完成，最优适应度: {self.best_score:.6f}")
        
        return self.best_score, self.get_best_parameters()

# 贝叶斯优化算法基本逻辑说明：
# 1. 初始化：使用拉丁超立方采样生成初始训练点
# 2. 高斯过程建模：用已观测点训练高斯过程，建立目标函数的概率模型
# 3. 采集函数：根据高斯过程的预测均值和方差，计算采集函数值
# 4. 优化采集函数：找到采集函数的最优点作为下一个评估点
# 5. 评估新点：计算新点的目标函数值，更新观测数据
# 6. 迭代：重复2-5步直到达到最大迭代次数
# 7. 优点：样本效率高，适合昂贵目标函数的优化
# 8. 特点：平衡探索和利用，具有理论保证的收敛性
# 9. 适用场景：目标函数评估代价高，需要少量样本找到最优解的情况
