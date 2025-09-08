# 优化算法通用工具函数模块
import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings


class OptimizationUtils:
    """
    优化算法通用工具类
    """
    
    @staticmethod
    def save_results(results, filename=None):
        """
        保存优化结果到JSON文件
        
        参数:
            results: 算法结果字典
            filename: 保存文件名，如果为None则使用时间戳
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        # 转换numpy数组为列表以便JSON序列化
        json_results = {}
        for alg_name, result in results.items():
            json_results[alg_name] = {
                'best_params': result['best_params'],
                'best_score': float(result['best_score']),
                'execution_time': float(result['execution_time']),
                'convergence_history': [float(x) for x in result['convergence_history']]
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=4, ensure_ascii=False)
        
        print(f"结果已保存到: {filename}")
        return filename
    
    @staticmethod
    def load_results(filename):
        """
        从JSON文件加载优化结果
        
        参数:
            filename: 结果文件名
            
        返回:
            results: 算法结果字典
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"结果已从 {filename} 加载")
            return results
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return None
        except json.JSONDecodeError:
            print(f"文件 {filename} 格式错误")
            return None
    
    @staticmethod
    def create_results_summary(results):
        """
        创建结果汇总表
        
        参数:
            results: 算法结果字典
            
        返回:
            df: pandas DataFrame格式的汇总表
        """
        summary_data = []
        
        for alg_name, result in results.items():
            summary_data.append({
                '算法': alg_name,
                'Best Score': result['best_score'],
                'Nugget': result['best_params']['nugget'],
                'Range': result['best_params']['range'],
                'Sill': result['best_params']['sill'],
                '执行时间(秒)': result['execution_time'],
                '收敛代数': len(result['convergence_history'])
            })
        
        df = pd.DataFrame(summary_data)
        
        # 按Best Score排序（假设分数越小越好）
        df = df.sort_values('Best Score').reset_index(drop=True)
        df['排名'] = df.index + 1
        
        # 重新排列列的顺序
        columns = ['排名', '算法', 'Best Score', 'Nugget', 'Range', 'Sill', '执行时间(秒)', '收敛代数']
        df = df[columns]
        
        return df
    
    @staticmethod
    def format_execution_time(seconds):
        """
        格式化执行时间
        
        参数:
            seconds: 执行时间（秒）
            
        返回:
            formatted_time: 格式化的时间字符串
        """
        if seconds < 60:
            return f"{seconds:.2f}秒"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{int(minutes)}分{remaining_seconds:.2f}秒"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            remaining_seconds = seconds % 60
            return f"{int(hours)}时{int(remaining_minutes)}分{remaining_seconds:.2f}秒"
    
    @staticmethod
    def calculate_improvement_percentage(baseline, optimized):
        """
        计算改进百分比
        
        参数:
            baseline: 基线值
            optimized: 优化后的值
            
        返回:
            improvement: 改进百分比
        """
        if baseline == 0:
            return 0
        return ((baseline - optimized) / abs(baseline)) * 100
    
    @staticmethod
    def suppress_warnings():
        """
        抑制常见的警告信息
        """
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
        warnings.filterwarnings('ignore', message='.*does not have valid gradient.*')
        warnings.filterwarnings('ignore', message='.*Optimization terminated successfully.*')
    
    @staticmethod
    def validate_parameters(params, bounds):
        """
        验证参数是否在给定范围内
        
        参数:
            params: 参数字典
            bounds: 参数边界字典
            
        返回:
            is_valid: 参数是否有效
            errors: 错误信息列表
        """
        errors = []
        
        for param_name, value in params.items():
            if param_name in bounds:
                min_val, max_val = bounds[param_name]
                if value < min_val or value > max_val:
                    errors.append(f"参数 {param_name} 值 {value} 超出范围 [{min_val}, {max_val}]")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def normalize_parameters(params, bounds):
        """
        将参数标准化到[0,1]范围
        
        参数:
            params: 参数字典
            bounds: 参数边界字典
            
        返回:
            normalized_params: 标准化后的参数字典
        """
        normalized_params = {}
        
        for param_name, value in params.items():
            if param_name in bounds:
                min_val, max_val = bounds[param_name]
                normalized_value = (value - min_val) / (max_val - min_val)
                normalized_params[param_name] = normalized_value
            else:
                normalized_params[param_name] = value
        
        return normalized_params
    
    @staticmethod
    def denormalize_parameters(normalized_params, bounds):
        """
        将标准化参数反标准化到原始范围
        
        参数:
            normalized_params: 标准化参数字典
            bounds: 参数边界字典
            
        返回:
            params: 原始范围的参数字典
        """
        params = {}
        
        for param_name, normalized_value in normalized_params.items():
            if param_name in bounds:
                min_val, max_val = bounds[param_name]
                value = min_val + normalized_value * (max_val - min_val)
                params[param_name] = value
            else:
                params[param_name] = normalized_value
        
        return params
    
    @staticmethod
    def generate_parameter_combinations(bounds, n_samples):
        """
        生成参数组合的拉丁超立方采样
        
        参数:
            bounds: 参数边界字典
            n_samples: 采样数量
            
        返回:
            param_combinations: 参数组合列表
        """
        from scipy.stats import qmc
        
        param_names = list(bounds.keys())
        n_params = len(param_names)
        
        # 生成拉丁超立方采样
        sampler = qmc.LatinHypercube(d=n_params, seed=42)
        samples = sampler.random(n=n_samples)
        
        param_combinations = []
        for sample in samples:
            params = {}
            for i, param_name in enumerate(param_names):
                min_val, max_val = bounds[param_name]
                params[param_name] = min_val + sample[i] * (max_val - min_val)
            param_combinations.append(params)
        
        return param_combinations
    
    @staticmethod
    def print_optimization_header(algorithm_name, max_iterations):
        """
        打印优化开始的标题信息
        
        参数:
            algorithm_name: 算法名称
            max_iterations: 最大迭代次数
        """
        print(f"\n{'='*60}")
        print(f"开始运行 {algorithm_name} 算法")
        print(f"最大迭代次数: {max_iterations}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
    
    @staticmethod
    def print_optimization_footer(algorithm_name, execution_time, best_score, best_params):
        """
        打印优化结束的总结信息
        
        参数:
            algorithm_name: 算法名称
            execution_time: 执行时间
            best_score: 最佳分数
            best_params: 最佳参数
        """
        print(f"\n{algorithm_name} 算法完成")
        print(f"执行时间: {OptimizationUtils.format_execution_time(execution_time)}")
        print(f"最佳分数: {best_score:.6f}")
        print(f"最佳参数: {best_params}")
        print(f"{'='*60}")
    
    @staticmethod
    def calculate_statistics(values):
        """
        计算数值统计信息
        
        参数:
            values: 数值列表
            
        返回:
            stats: 统计信息字典
        """
        values = np.array(values)
        
        stats = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
        
        return stats


class ConvergenceAnalyzer:
    """
    收敛分析工具类
    """
    
    @staticmethod
    def analyze_convergence(convergence_history, window_size=10):
        """
        分析收敛性
        
        参数:
            convergence_history: 收敛历史
            window_size: 滑动窗口大小
            
        返回:
            analysis: 收敛分析结果
        """
        history = np.array(convergence_history)
        
        # 计算改进率
        improvements = []
        for i in range(1, len(history)):
            if history[i-1] != 0:
                improvement = (history[i-1] - history[i]) / abs(history[i-1])
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        # 滑动窗口平均
        if len(history) >= window_size:
            window_averages = []
            for i in range(window_size, len(history)):
                window_avg = np.mean(history[i-window_size:i])
                window_averages.append(window_avg)
        else:
            window_averages = []
        
        # 判断是否收敛
        converged = False
        convergence_iteration = None
        if len(improvements) >= window_size:
            recent_improvements = improvements[-window_size:]
            if all(imp < 0.001 for imp in recent_improvements):  # 最近改进都小于0.1%
                converged = True
                convergence_iteration = len(history) - window_size
        
        analysis = {
            'converged': converged,
            'convergence_iteration': convergence_iteration,
            'total_iterations': len(history),
            'final_score': history[-1] if len(history) > 0 else None,
            'best_score': np.min(history) if len(history) > 0 else None,
            'improvements': improvements,
            'window_averages': window_averages,
            'convergence_rate': np.mean(improvements) if improvements else 0
        }
        
        return analysis
    
    @staticmethod
    def detect_premature_convergence(convergence_history, threshold=1e-6, min_iterations=20):
        """
        检测过早收敛
        
        参数:
            convergence_history: 收敛历史
            threshold: 收敛阈值
            min_iterations: 最小迭代次数
            
        返回:
            is_premature: 是否过早收敛
            stagnation_start: 停滞开始的迭代次数
        """
        history = np.array(convergence_history)
        
        if len(history) < min_iterations:
            return False, None
        
        # 查找连续多次改进极小的情况
        for i in range(min_iterations, len(history)):
            if i >= 10:  # 至少检查10次迭代
                recent_variance = np.var(history[i-10:i])
                if recent_variance < threshold:
                    return True, i-10
        
        return False, None
