# 克里金插值参数优化算法比较和可视化
# 常量定义
DEFAULT_DATA_PATH = './data/real_data/地层坐标.xlsx'
DEFAULT_TARGET_LAYER = '松散层'
DEFAULT_GRID_RESOLUTION = 100
FIGURE_SIZE = (20, 15)
COMPARISON_ALGORITHMS = ['GA', 'PSO', 'ACO', 'DE', 'Bayesian']
GENERATIONS = 400
TRAIN_RATIO = 0.8
RANDOM_SEED = 0
OBJECTION_FUNCTION = 'rmse'
VERBOSE=True
VARIOGRAM_MODEL = 'spherical'

import numpy as np
import matplotlib.pyplot as plt
import warnings
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import pdist, squareform
import time
import json

# 过滤优化过程中的警告
warnings.filterwarnings('ignore', category=Warning, module='sklearn')
warnings.filterwarnings('ignore', message='.*convergence.*', category=Warning)

# 导入优化器
from genetic_algorithm_optimizer import GAKrigeOptimizer
from particle_swarm_optimizer import PSOKrigeOptimizer
from ant_colony_optimizer import ACOKrigeOptimizer
from differential_evolution_optimizer import DEKrigeOptimizer
from bayesian_optimizer import BayesianKrigeOptimizer

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class KrigeOptimizationComparison:
    """
    克里金插值参数优化算法比较类
    """
    
    def __init__(self, data_path=DEFAULT_DATA_PATH, target_layer=DEFAULT_TARGET_LAYER):
        """
        初始化比较器
        
        参数:
            data_path: 数据文件路径
            target_layer: 目标地层名称
        """
        self.data_path = data_path
        self.target_layer = target_layer
        self.results = {}
        
    def run_algorithm_comparison(self, algorithms=COMPARISON_ALGORITHMS, 
                                train_ratio=TRAIN_RATIO, random_seed=RANDOM_SEED, 
                                objective=OBJECTION_FUNCTION, verbose=VERBOSE):
        """
        运行算法比较
        
        参数:
            algorithms: 要比较的算法列表
            train_ratio: 训练集比例
            random_seed: 随机种子
            objective: 目标函数类型
            verbose: 是否打印详细信息
            
        返回:
            results: 比较结果字典
        """
        # 初始化优化器
        optimizers = {
            'GA': GAKrigeOptimizer(generations=GENERATIONS, population_size=30),
            'PSO': PSOKrigeOptimizer(iterations=GENERATIONS, n_particles=20),
            'ACO': ACOKrigeOptimizer(iterations=GENERATIONS, n_ants=50),
            'DE': DEKrigeOptimizer(generations=GENERATIONS, population_size=30),
            'Bayesian': BayesianKrigeOptimizer(iterations=50, init_samples=10)
        }
        
        self.results = {}
        
        for alg_name in algorithms:
            if alg_name not in optimizers:
                print(f"警告: 未知算法 {alg_name}, 跳过")
                continue
                
            print(f"\n{'='*50}")
            print(f"运行 {alg_name} 算法...")
            print(f"{'='*50}")
            
            optimizer = optimizers[alg_name]
            
            # 加载和分割数据
            train_data, test_data = optimizer.load_and_split_data(
                self.data_path, self.target_layer, train_ratio, random_seed
            )
            
            # 定义参数边界
            x_train, y_train, z_train = train_data
            bounds = optimizer.define_parameter_bounds(x_train, y_train, z_train)
            
            # 运行优化
            start_time = time.time()
            best_score, best_params = optimizer.optimize(
                train_data, test_data, bounds, objective, verbose
            )
            end_time = time.time()
            
            # 存储结果
            self.results[alg_name] = {
                'best_score': best_score,
                'best_params': best_params,
                'runtime': end_time - start_time,
                'train_data': train_data,
                'test_data': test_data,
                'bounds': bounds
            }
            
            print(f"{alg_name} 完成，用时: {end_time - start_time:.2f}秒")
            print(f"最优参数: {best_params}")
        
        return self.results
    
    def plot_semivariogram_comparison(self, save_path=None):
        """
        绘制半变异函数比较图
        
        参数:
            save_path: 图片保存路径
        """
        if not self.results:
            print("请先运行算法比较")
            return
        
        # 使用第一个算法的数据
        first_alg = list(self.results.keys())[0]
        x_train, y_train, z_train = self.results[first_alg]['train_data']
        
        # 计算实验半方差
        coords = np.vstack((x_train, y_train)).T
        dists = squareform(pdist(coords))
        semivariances = 0.5 * (z_train[:, None] - z_train[None, :]) ** 2
        triu_indices = np.triu_indices_from(dists, k=1)
        h = dists[triu_indices]
        gamma = semivariances[triu_indices]
        
        # 分组计算
        nlags = 15
        bins = np.linspace(h.min(), h.max(), nlags + 1)
        bin_indices = np.digitize(h, bins)
        bin_centers = []
        gamma_means = []
        
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.any(mask):
                bin_centers.append(h[mask].mean())
                gamma_means.append(gamma[mask].mean())
        
        bin_centers = np.array(bin_centers)
        gamma_means = np.array(gamma_means)
        
        # 绘图
        plt.figure(figsize=(15, 10))
        
        # 实验半方差
        plt.scatter(bin_centers, gamma_means, color='black', s=50, 
                   label='实验半方差', zorder=3)
        
        # 理论变差函数
        h_fit = np.linspace(0, h.max(), 200)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (alg_name, result) in enumerate(self.results.items()):
            params = result['best_params']
            nugget = params['nugget']
            range_val = params['range']
            sill = params['sill']
            
            # 球状模型
            gamma_fit = self._spherical_model(h_fit, nugget, range_val, sill)
            
            plt.plot(h_fit, gamma_fit, color=colors[i % len(colors)], 
                    linewidth=2, label=f'{alg_name} 优化')
        
        plt.xlabel('距离 h')
        plt.ylabel('半方差 γ(h)')
        plt.title('不同优化算法的半变异函数比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interpolation_comparison(self, grid_resolution=DEFAULT_GRID_RESOLUTION, 
                                    save_path=None):
        """
        绘制插值结果比较图
        
        参数:
            grid_resolution: 网格分辨率
            save_path: 图片保存路径
        """
        if not self.results:
            print("请先运行算法比较")
            return
        
        # 使用第一个算法的数据
        first_alg = list(self.results.keys())[0]
        x_train, y_train, z_train = self.results[first_alg]['train_data']
        x_test, y_test, z_test = self.results[first_alg]['test_data']
        
        # 创建插值网格
        x_all = np.concatenate([x_train, x_test])
        y_all = np.concatenate([y_train, y_test])
        
        grid_x = np.linspace(x_all.min(), x_all.max(), grid_resolution)
        grid_y = np.linspace(y_all.min(), y_all.max(), grid_resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        
        # 计算每个算法的插值结果
        n_algs = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
        axes = axes.flatten()
        
        # 默认克里金插值
        ok_default = OrdinaryKriging(
            x_train, y_train, z_train,
            variogram_model=VARIOGRAM_MODEL,
            verbose=False,
            enable_plotting=False
        )
        z_default, ss_default = ok_default.execute('grid', grid_x, grid_y)
        
        # 绘制默认插值
        cs = axes[0].contourf(grid_xx, grid_yy, z_default, levels=20, cmap='viridis')
        axes[0].scatter(x_train, y_train, c='white', s=30, edgecolor='black')
        axes[0].scatter(x_test, y_test, c='red', s=30, marker='x')
        axes[0].set_title('默认参数插值')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        plt.colorbar(cs, ax=axes[0])
        
        # 绘制优化后的插值
        for i, (alg_name, result) in enumerate(self.results.items()):
            if i >= 5:  # 最多显示5个算法
                break
                
            params = result['best_params']
            variogram_params = [params['nugget'], params['range'], params['sill']]
            
            ok_opt = OrdinaryKriging(
                x_train, y_train, z_train,
                variogram_model=VARIOGRAM_MODEL,
                variogram_parameters=variogram_params,
                verbose=False,
                enable_plotting=False
            )
            z_opt, ss_opt = ok_opt.execute('grid', grid_x, grid_y)
            
            cs = axes[i + 1].contourf(grid_xx, grid_yy, z_opt, levels=20, cmap='viridis')
            axes[i + 1].scatter(x_train, y_train, c='white', s=30, edgecolor='black')
            axes[i + 1].scatter(x_test, y_test, c='red', s=30, marker='x')
            axes[i + 1].set_title(f'{alg_name} 优化插值')
            axes[i + 1].set_xlabel('X')
            axes[i + 1].set_ylabel('Y')
            plt.colorbar(cs, ax=axes[i + 1])
        
        # 隐藏多余的子图
        for j in range(len(self.results) + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_comparison(self, save_path=None):
        """
        绘制收敛性比较图（需要算法返回收敛历史）
        
        参数:
            save_path: 图片保存路径
        """
        # 这里需要修改优化器以返回收敛历史
        # 暂时显示最终结果比较
        self.plot_results_summary(save_path)
    
    def plot_results_summary(self, save_path=None):
        """
        绘制结果总结图
        
        参数:
            save_path: 图片保存路径
        """
        if not self.results:
            print("请先运行算法比较")
            return
        
        alg_names = list(self.results.keys())
        scores = [self.results[alg]['best_score'] for alg in alg_names]
        runtimes = [self.results[alg]['runtime'] for alg in alg_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 适应度比较
        bars1 = ax1.bar(alg_names, scores, color=['red', 'blue', 'green', 'orange', 'purple'])
        ax1.set_ylabel('RMSE')
        ax1.set_title('算法性能比较')
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱状图上显示数值
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom')
        
        # 运行时间比较
        bars2 = ax2.bar(alg_names, runtimes, color=['red', 'blue', 'green', 'orange', 'purple'])
        ax2.set_ylabel('运行时间 (秒)')
        ax2.set_title('算法效率比较')
        ax2.tick_params(axis='x', rotation=45)
        
        # 在柱状图上显示数值
        for bar, runtime in zip(bars2, runtimes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{runtime:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_results(self, file_path):
        """
        导出结果到JSON文件
        
        参数:
            file_path: 导出文件路径
        """
        if not self.results:
            print("请先运行算法比较")
            return
        
        # 转换numpy数组为列表，以便JSON序列化
        export_data = {}
        for alg_name, result in self.results.items():
            export_data[alg_name] = {
                'best_score': float(result['best_score']),
                'best_params': {
                    k: float(v) for k, v in result['best_params'].items()
                },
                'runtime': float(result['runtime'])
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"结果已导出到: {file_path}")
    
    def _spherical_model(self, h, nugget, range_val, sill):
        """
        球状模型变差函数
        
        参数:
            h: 距离
            nugget: 块金效应
            range_val: 变程
            sill: 基台值
            
        返回:
            gamma: 半方差值
        """
        h = np.array(h)
        return np.piecewise(
            h,
            [h <= range_val, h > range_val],
            [
                lambda x: nugget + (sill - nugget) * (1.5 * x / range_val - 0.5 * (x / range_val) ** 3),
                lambda x: sill
            ]
        )


def main():
    """
    主函数：运行算法比较示例
    """
    # 创建比较器
    comparator = KrigeOptimizationComparison()
    
    # 运行算法比较
    print("开始克里金插值参数优化算法比较...")
    results = comparator.run_algorithm_comparison(
        algorithms=COMPARISON_ALGORITHMS,  # 可以调整要比较的算法
        train_ratio=0.7,
        random_seed=42,
        objective='rmse',
        verbose=True
    )
    
    # 绘制比较图
    print("\n生成比较图...")
    comparator.plot_semivariogram_comparison()
    comparator.plot_interpolation_comparison()
    comparator.plot_results_summary()
    
    # 导出结果
    comparator.export_results('optimization_results.json')
    
    print("\n算法比较完成！")


if __name__ == '__main__':
    main()

# 使用说明：
# 1. 确保所有优化器文件和数据文件在正确位置
# 2. 根据需要调整算法参数和比较设置
# 3. 运行此文件即可进行算法比较和可视化
# 4. 结果会以图表形式显示，并可导出为JSON文件
