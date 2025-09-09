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
from sklearn.cluster import DBSCAN
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
        评估参数组合的适应度 - 使用按井留一验证
        
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
            
            # 合并所有数据
            x_all = np.concatenate([x_train, x_test])
            y_all = np.concatenate([y_train, y_test])
            z_all = np.concatenate([z_train, z_test])
            
            # 通过空间聚类识别"钻孔"组（假设同一钻孔的点在空间上接近）
            from sklearn.cluster import DBSCAN
            coords = np.column_stack([x_all, y_all])
            
            # 计算合适的聚类距离（基于点间距离的分位数）
            from scipy.spatial.distance import pdist
            distances = pdist(coords)
            eps = np.percentile(distances, 10)  # 使用10%分位数作为聚类半径
            
            # 执行聚类，每个cluster代表一个"钻孔"
            clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)
            borehole_labels = clustering.labels_
            
            # 获取唯一的钻孔ID（排除噪声点-1）
            unique_boreholes = np.unique(borehole_labels)
            unique_boreholes = unique_boreholes[unique_boreholes >= 0]
            
            if len(unique_boreholes) < 2:
                # 如果钻孔数量不足，回退到传统验证
                return self._traditional_validation(nugget, range_val, sill,
                                                  x_train, y_train, z_train,
                                                  x_test, y_test, z_test, objective)
            
            # 按井留一验证
            all_errors = []
            successful_validations = 0
            
            for test_borehole in unique_boreholes:
                # 获取测试钻孔的数据点
                test_mask = borehole_labels == test_borehole
                x_test_bh = x_all[test_mask]
                y_test_bh = y_all[test_mask]
                z_test_bh = z_all[test_mask]
                
                # 获取训练数据（其他所有钻孔）
                train_mask = borehole_labels != test_borehole
                x_train_bh = x_all[train_mask]
                y_train_bh = y_all[train_mask]
                z_train_bh = z_all[train_mask]
                
                # 确保有足够的训练数据
                if len(x_train_bh) < 3:
                    continue
                
                try:
                    # 创建克里金模型
                    ok = OrdinaryKriging(
                        x_train_bh, y_train_bh, z_train_bh,
                        variogram_model=DEFAULT_VARIOGRAM_MODEL,
                        variogram_parameters=[nugget, range_val, sill],
                        verbose=False,
                        enable_plotting=False
                    )
                    
                    # 预测测试钻孔
                    z_pred_bh, _ = ok.execute('points', x_test_bh, y_test_bh)
                    
                    # 计算该钻孔的误差
                    if objective == 'rmse':
                        error = np.sqrt(np.mean((z_test_bh - z_pred_bh) ** 2))
                    elif objective == 'mae':
                        error = np.mean(np.abs(z_test_bh - z_pred_bh))
                    elif objective == 'likelihood':
                        residuals = z_test_bh - z_pred_bh
                        error = np.sum(residuals**2)
                    else:
                        error = np.mean((z_test_bh - z_pred_bh) ** 2)
                    
                    all_errors.append(error)
                    successful_validations += 1
                    
                except Exception:
                    # 如果某个钻孔验证失败，给予惩罚分数
                    all_errors.append(LARGE_ERROR_VALUE * 0.1)
            
            if successful_validations == 0:
                return LARGE_ERROR_VALUE
            
            # 计算平均误差
            mean_error = np.mean(all_errors)
            
            # 如果成功验证的钻孔数量太少，给予惩罚
            if successful_validations < len(unique_boreholes) * 0.5:
                mean_error *= (2.0 - successful_validations / len(unique_boreholes))
            
            return mean_error
            
        except Exception as e:
            return LARGE_ERROR_VALUE
    
    def _traditional_validation(self, nugget, range_val, sill,
                               x_train, y_train, z_train,
                               x_test, y_test, z_test, objective):
        """
        传统验证方法（备用）
        """
        try:
            ok = OrdinaryKriging(
                x_train, y_train, z_train,
                variogram_model=DEFAULT_VARIOGRAM_MODEL,
                variogram_parameters=[nugget, range_val, sill],
                verbose=False,
                enable_plotting=False
            )
            
            z_pred, _ = ok.execute('points', x_test, y_test)
            
            if objective == 'rmse':
                error = np.sqrt(np.mean((z_test - z_pred) ** 2))
            elif objective == 'mae':
                error = np.mean(np.abs(z_test - z_pred))
            elif objective == 'likelihood':
                residuals = z_test - z_pred
                error = np.sum(residuals**2)
            else:
                error = np.mean((z_test - z_pred) ** 2)
            
            trend_corr, _ = spearmanr(z_pred, z_test)
            if np.isnan(trend_corr):
                trend_corr = 0
            penalty = (1 - abs(trend_corr)) ** 2
            
            return error * (1 + PENALTY_WEIGHT * penalty)
            
        except Exception:
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
    
    def get_default_kriging_parameters(self, train_data):
        """
        获取默认克里金插值的参数
        
        参数:
            train_data: 训练数据 (x_train, y_train, z_train)
            
        返回:
            default_params: 默认参数字典
        """
        x_train, y_train, z_train = train_data
        
        try:
            # 创建默认克里金模型以获取自动拟合的参数
            ok_default = OrdinaryKriging(
                x_train, y_train, z_train,
                variogram_model=DEFAULT_VARIOGRAM_MODEL,
                verbose=False,
                enable_plotting=False
            )
            
            # 获取自动拟合的变差函数参数
            if hasattr(ok_default, 'variogram_function_parameters'):
                params = ok_default.variogram_function_parameters
                default_nugget = params[0] if len(params) > 0 else 0.0
                default_range = params[1] if len(params) > 1 else 1000.0
                default_sill = params[2] if len(params) > 2 else 1.0
            else:
                # 如果无法获取参数，使用经验值
                var_z = np.var(z_train)
                default_nugget = var_z * 0.1
                default_range = np.percentile(np.sqrt((x_train[:, None] - x_train)**2 + 
                                                    (y_train[:, None] - y_train)**2), 50)
                default_sill = var_z * 0.9
            
            return {
                'nugget': default_nugget,
                'range': default_range,
                'sill': default_sill,
                'method': 'default_kriging'
            }
            
        except Exception as e:
            print(f"获取默认克里金参数失败: {e}")
            # 返回经验默认值
            var_z = np.var(z_train)
            return {
                'nugget': var_z * 0.1,
                'range': 1000.0,
                'sill': var_z * 0.9,
                'method': 'default_kriging'
            }
    
    def calculate_error_metrics(self, nugget, range_val, sill, train_data, test_data):
        """
        计算给定参数的详细误差指标
        
        参数:
            nugget, range_val, sill: 变差函数参数
            train_data: 训练数据
            test_data: 测试数据
            
        返回:
            metrics: 误差指标字典
        """
        x_train, y_train, z_train = train_data
        x_test, y_test, z_test = test_data
        
        try:
            # 创建克里金模型
            ok = OrdinaryKriging(
                x_train, y_train, z_train,
                variogram_model=DEFAULT_VARIOGRAM_MODEL,
                variogram_parameters=[nugget, range_val, sill],
                verbose=False,
                enable_plotting=False
            )
            
            # 预测
            z_pred, z_var = ok.execute('points', x_test, y_test)
            
            # 计算各种误差指标
            residuals = z_pred - z_test
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(np.abs(residuals))
            bias = np.mean(residuals)
            r2 = 1 - np.sum(residuals**2) / np.sum((z_test - np.mean(z_test))**2)
            max_error = np.max(np.abs(residuals))
            
            # 趋势一致性
            trend_corr, _ = spearmanr(z_pred, z_test)
            if np.isnan(trend_corr):
                trend_corr = 0
            
            return {
                'rmse': rmse,
                'mae': mae,
                'bias': bias,
                'r2': r2,
                'max_error': max_error,
                'trend_correlation': trend_corr,
                'mean_variance': np.mean(z_var),
                'parameters': {
                    'nugget': nugget,
                    'range': range_val,
                    'sill': sill
                },
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'rmse': np.inf,
                'mae': np.inf,
                'bias': np.inf,
                'r2': -np.inf,
                'max_error': np.inf,
                'trend_correlation': 0,
                'mean_variance': np.inf,
                'parameters': {
                    'nugget': nugget,
                    'range': range_val,
                    'sill': sill
                },
                'status': 'error',
                'error_message': str(e)
            }
    
    def compare_all_parameters(self, train_data, test_data, optimized_params=None):
        """
        比较默认克里金参数和优化后参数的误差
        
        参数:
            train_data: 训练数据
            test_data: 测试数据  
            optimized_params: 优化后的参数字典（可选）
            
        返回:
            comparison_results: 比较结果字典
        """
        print("\n" + "="*80)
        print("参数误差对比分析")
        print("="*80)
        
        results = {}
        
        # 1. 计算默认克里金参数的误差
        print("1. 计算默认克里金参数误差...")
        default_params = self.get_default_kriging_parameters(train_data)
        default_metrics = self.calculate_error_metrics(
            default_params['nugget'], 
            default_params['range'], 
            default_params['sill'],
            train_data, test_data
        )
        default_metrics['method'] = 'Default Kriging'
        results['default'] = default_metrics
        
        # 2. 如果有优化后的参数，计算其误差
        if optimized_params is not None:
            print("2. 计算优化参数误差...")
            optimized_metrics = self.calculate_error_metrics(
                optimized_params['nugget'],
                optimized_params['range'], 
                optimized_params['sill'],
                train_data, test_data
            )
            optimized_metrics['method'] = getattr(optimized_params, 'method', 'Optimized')
            results['optimized'] = optimized_metrics
        
        # 3. 如果有当前最优参数，也计算其误差
        if (self.best_nugget is not None and 
            self.best_range is not None and 
            self.best_sill is not None):
            print("3. 计算当前最优参数误差...")
            best_metrics = self.calculate_error_metrics(
                self.best_nugget,
                self.best_range,
                self.best_sill,
                train_data, test_data
            )
            best_metrics['method'] = 'Current Best'
            results['best'] = best_metrics
        
        # 4. 打印对比结果
        self._print_comparison_table(results)
        
        return results
    
    def _print_comparison_table(self, results):
        """打印对比结果表格"""
        print("\n误差指标对比表:")
        print("-"*80)
        
        # 表头
        print(f"{'Method':<15} {'RMSE':<10} {'MAE':<10} {'R²':<8} {'Bias':<10} {'Max Err':<10}")
        print("-"*80)
        
        # 各方法的结果
        for key, metrics in results.items():
            if metrics['status'] == 'success':
                method = metrics['method']
                rmse = metrics['rmse']
                mae = metrics['mae']
                r2 = metrics['r2']
                bias = metrics['bias']
                max_err = metrics['max_error']
                
                print(f"{method:<15} {rmse:<10.4f} {mae:<10.4f} {r2:<8.4f} {bias:<10.4f} {max_err:<10.4f}")
            else:
                print(f"{key:<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")
        
        print("-"*80)
        
        # 参数对比
        print("\n参数对比:")
        print("-"*60)
        print(f"{'Method':<15} {'Nugget':<12} {'Range':<12} {'Sill':<12}")
        print("-"*60)
        
        for key, metrics in results.items():
            if metrics['status'] == 'success':
                method = metrics['method']
                params = metrics['parameters']
                print(f"{method:<15} {params['nugget']:<12.6f} {params['range']:<12.2f} {params['sill']:<12.6f}")
        
        print("-"*60)
        
        # 性能改进分析
        if len(results) > 1 and 'default' in results:
            print("\n性能改进分析:")
            print("-"*40)
            default_rmse = results['default']['rmse']
            
            for key, metrics in results.items():
                if key != 'default' and metrics['status'] == 'success':
                    improvement = ((default_rmse - metrics['rmse']) / default_rmse) * 100
                    print(f"{metrics['method']} vs Default: {improvement:+.2f}% RMSE改进")
        
        print("="*80)
    
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
