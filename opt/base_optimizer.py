# 克里金插值参数优化基类
# 常量定义
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_RANDOM_SEED = 0
DEFAULT_VARIOGRAM_MODEL = 'spherical'
PENALTY_WEIGHT = 1.0
LARGE_ERROR_VALUE = 1e6

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from abc import ABC, abstractmethod


class BaseKrigeOptimizer(ABC):
    """
    克里金插值参数优化基类
    提供数据处理、参数空间定义、适应度评估等公共功能
    """
    
    def __init__(self):
        """初始化优化器"""
        self.best_nugget = None
        self.best_range = None
        self.best_sill = None
        self.best_score = float('inf')
        
    def load_and_split_data(self, data_path, target_layer="松散层", 
                           train_ratio=DEFAULT_TRAIN_RATIO, 
                           random_seed=DEFAULT_RANDOM_SEED):
        """
        加载数据并分割为训练集和测试集
        
        参数:
            data_path: 数据文件路径
            target_layer: 目标地层名称
            train_ratio: 训练集比例
            random_seed: 随机种子
            
        返回:
            train_data: (x_train, y_train, z_train)
            test_data: (x_test, y_test, z_test)
        """
        np.random.seed(random_seed)
        df = pd.read_excel(data_path)
        layer_df = df[df["地层名称"] == target_layer]
        
        x = layer_df["x"].values.astype(np.float64)
        y = layer_df["y"].values.astype(np.float64)
        z = layer_df["z"].values.astype(np.float64)
        
        n_points = len(x)
        n_train = int(n_points * train_ratio)
        train_idx = np.random.choice(n_points, n_train, replace=False)
        test_idx = np.setdiff1d(np.arange(n_points), train_idx)
        
        print(f"数据分割: {len(train_idx)} 训练点, {len(test_idx)} 测试点")
        
        train_data = (x[train_idx], y[train_idx], z[train_idx])
        test_data = (x[test_idx], y[test_idx], z[test_idx])
        
        return train_data, test_data
    
    def define_parameter_bounds(self, x, y, z):
        """
        定义参数搜索边界
        
        参数:
            x, y, z: 坐标和属性值
            
        返回:
            bounds: {'nugget': (min, max), 'range': (min, max), 'sill': (min, max)}
        """
        # 计算数据统计量
        var_z = np.var(z)
        domain_size = max(x.max() - x.min(), y.max() - y.min())
        
        # 计算点间距离分布
        coords = np.vstack([x, y]).T
        dists = pdist(coords)
        min_dist = np.percentile(dists, 5)
        max_dist = np.percentile(dists, 95)
        
        # 定义参数边界
        bounds = {
            'nugget': (0, var_z * 0.8),
            'range': (min_dist, min(max_dist, domain_size * 1.5)),
            'sill': (var_z * 0.1, var_z * 3.0)
        }
        
        return bounds
    
    def evaluate_fitness(self, nugget, range_val, sill, 
                        x_train, y_train, z_train, 
                        x_test, y_test, z_test,
                        objective='rmse'):
        """
        评估参数组合的适应度
        
        参数:
            nugget, range_val, sill: 变差函数参数
            x_train, y_train, z_train: 训练数据
            x_test, y_test, z_test: 测试数据
            objective: 目标函数类型 ('rmse', 'mae', 'likelihood')
            
        返回:
            fitness: 适应度值（越小越好）
        """
        try:
            # 参数有效性检查
            if sill <= nugget or range_val <= 0:
                return LARGE_ERROR_VALUE
                
            # 创建克里金模型
            ok = OrdinaryKriging(
                x_train, y_train, z_train,
                variogram_model=DEFAULT_VARIOGRAM_MODEL,
                variogram_parameters=[nugget, range_val, sill],
                verbose=False,
                enable_plotting=False
            )
            
            # 预测
            z_pred, _ = ok.execute('points', x_test, y_test)
            
            # 计算误差
            if objective == 'rmse':
                error = np.sqrt(np.mean((z_test - z_pred) ** 2))
            elif objective == 'mae':
                error = np.mean(np.abs(z_test - z_pred))
            elif objective == 'likelihood':
                # 简化的似然函数（负对数似然）
                residuals = z_test - z_pred
                error = np.sum(residuals**2)
            else:
                error = np.mean((z_test - z_pred) ** 2)
            
            # 趋势一致性惩罚
            trend_corr, _ = spearmanr(z_pred, z_test)
            if np.isnan(trend_corr):
                trend_corr = 0
            penalty = (1 - abs(trend_corr)) ** 2
            
            return error * (1 + PENALTY_WEIGHT * penalty)
            
        except Exception as e:
            return LARGE_ERROR_VALUE
    
    def get_best_parameters(self):
        """
        获取最优参数
        
        返回:
            dict: 包含最优参数的字典
        """
        return {
            'nugget': self.best_nugget,
            'range': self.best_range,
            'sill': self.best_sill,
            'score': self.best_score
        }
    
    @abstractmethod
    def optimize(self, train_data, test_data, bounds, **kwargs):
        """
        抽象方法：执行优化
        
        参数:
            train_data: 训练数据
            test_data: 测试数据
            bounds: 参数边界
            **kwargs: 其他参数
            
        返回:
            best_score: 最优适应度
            best_params: 最优参数字典
        """
        pass

# 算法基本逻辑说明：
# 1. 基类提供了数据处理、参数边界定义、适应度评估等公共功能
# 2. 适应度评估结合了预测误差和趋势一致性，确保插值结果的合理性
# 3. 参数边界根据数据特征自动确定，避免不合理的参数组合
# 4. 支持多种目标函数（RMSE、MAE、似然函数），满足不同需求
