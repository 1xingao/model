# 克里金插值参数优化算法比较和可视化（精简版）
# 使用模块化设计，主要功能已拆分到独立文件中

# 常量定义
DEFAULT_DATA_PATH = './data/real_data/地层坐标.xlsx'
DEFAULT_TARGET_LAYER = '松散层'
COMPARISON_ALGORITHMS = ['GA', 'PSO', 'ACO', 'DE', 'Bayesian']
GENERATIONS = 400
TRAIN_RATIO = 0.8
RANDOM_SEED = 0
OBJECTION_FUNCTION = 'rmse'
VERBOSE = True

import time

# 导入优化器
from genetic_algorithm_optimizer import GAKrigeOptimizer
from particle_swarm_optimizer import PSOKrigeOptimizer
from ant_colony_optimizer import ACOKrigeOptimizer
from differential_evolution_optimizer import DEKrigeOptimizer
from bayesian_optimizer import BayesianKrigeOptimizer

# 导入新的模块化组件
from visualization import KrigeVisualization
from comparison_analysis import KrigeComparison
from utils import OptimizationUtils, ConvergenceAnalyzer


class KrigeOptimizationComparison:
    """
    克里金插值参数优化算法比较类（精简版）
    
    主要功能：
    1. 运行多种优化算法
    2. 协调各模块进行分析和可视化
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
        
        # 初始化算法字典
        self.algorithms = {
            'GA': GAKrigeOptimizer(data_path, target_layer),
            'PSO': PSOKrigeOptimizer(data_path, target_layer),
            'ACO': ACOKrigeOptimizer(data_path, target_layer),
            'DE': DEKrigeOptimizer(data_path, target_layer),
            'Bayesian': BayesianKrigeOptimizer(data_path, target_layer)
        }
        
        # 初始化工具组件
        self.visualizer = KrigeVisualization()
        self.comparator = KrigeComparison()
        self.utils = OptimizationUtils()
        self.convergence_analyzer = ConvergenceAnalyzer()
        
        # 抑制警告
        self.utils.suppress_warnings()
    
    def run_algorithm_comparison(self, algorithms=None, max_iterations=GENERATIONS, 
                               save_results=True, results_filename=None):
        """
        运行算法比较
        
        参数:
            algorithms: 要比较的算法列表，默认全部算法
            max_iterations: 最大迭代次数
            save_results: 是否保存结果
            results_filename: 结果文件名
            
        返回:
            results: 算法结果字典
        """
        if algorithms is None:
            algorithms = COMPARISON_ALGORITHMS
        
        results = {}
        
        print(f"\n{'='*80}")
        print("克里金插值参数优化算法比较")
        print(f"数据路径: {self.data_path}")
        print(f"目标地层: {self.target_layer}")
        print(f"比较算法: {', '.join(algorithms)}")
        print(f"最大迭代次数: {max_iterations}")
        print(f"{'='*80}")
        
        for alg_name in algorithms:
            if alg_name not in self.algorithms:
                print(f"警告: 算法 {alg_name} 不存在，跳过")
                continue
            
            optimizer = self.algorithms[alg_name]
            
            # 打印算法开始信息
            self.utils.print_optimization_header(alg_name, max_iterations)
            
            # 运行优化
            start_time = time.time()
            best_params, best_score, convergence_history = optimizer.optimize(max_iterations=max_iterations)
            execution_time = time.time() - start_time
            
            # 保存结果
            results[alg_name] = {
                'best_params': best_params,
                'best_score': best_score,
                'execution_time': execution_time,
                'convergence_history': convergence_history,
                'train_data': (optimizer.x_train, optimizer.y_train, optimizer.z_train),
                'test_data': (optimizer.x_test, optimizer.y_test, optimizer.z_test)
            }
            
            # 打印算法结束信息
            self.utils.print_optimization_footer(alg_name, execution_time, best_score, best_params)
            
            # 分析收敛性
            convergence_analysis = self.convergence_analyzer.analyze_convergence(convergence_history)
            if convergence_analysis['converged']:
                print(f"算法在第 {convergence_analysis['convergence_iteration']} 代收敛")
            else:
                print(f"算法在 {max_iterations} 代内未完全收敛")
        
        # 保存结果
        if save_results:
            self.utils.save_results(results, results_filename)
        
        # 打印总结
        self._print_comparison_summary(results)
        
        return results
    
    def visualize_results(self, results, save_plots=False, plots_dir='./plots/'):
        """
        可视化结果
        
        参数:
            results: 算法结果字典
            save_plots: 是否保存图片
            plots_dir: 图片保存目录
        """
        if not results:
            print("没有结果可用于可视化")
            return
        
        print(f"\n{'='*60}")
        print("开始可视化分析")
        print(f"{'='*60}")
        
        # 1. 可视化收敛过程
        convergence_save_path = f"{plots_dir}convergence_comparison.png" if save_plots else None
        self.visualizer.plot_convergence_comparison(results, convergence_save_path)
        
        # 2. 可视化参数分布
        params_save_path = f"{plots_dir}parameters_comparison.png" if save_plots else None
        self.visualizer.plot_parameters_comparison(results, params_save_path)
        
        # 3. 可视化半变异函数
        semivariogram_save_path = f"{plots_dir}semivariogram_comparison.png" if save_plots else None
        # 使用第一个算法的数据
        first_result = list(results.values())[0]
        x_train, y_train, z_train = first_result['train_data']
        self.visualizer.plot_semivariogram_comparison(x_train, y_train, z_train, results, semivariogram_save_path)
        
        # 4. 可视化插值对比
        interpolation_save_path = f"{plots_dir}interpolation_comparison.png" if save_plots else None
        self.visualizer.plot_interpolation_comparison(results, interpolation_save_path)
        
        # 5. 可视化结果总结
        summary_save_path = f"{plots_dir}results_summary.png" if save_plots else None
        self.visualizer.plot_results_summary(results, summary_save_path)
        
        print("可视化完成")
    
    def analyze_vs_default(self, results, algorithm_name=None, save_plots=False, plots_dir='./plots/'):
        """
        分析优化结果与默认参数的对比
        
        参数:
            results: 算法结果字典
            algorithm_name: 指定算法名称
            save_plots: 是否保存图片
            plots_dir: 图片保存目录
        """
        if not results:
            print("没有结果可用于分析")
            return
        
        print(f"\n{'='*60}")
        print("开始默认参数 vs 优化参数对比分析")
        print(f"{'='*60}")
        
        if algorithm_name:
            # 分析单个算法
            comparison_save_path = f"{plots_dir}default_vs_{algorithm_name.lower()}.png" if save_plots else None
            comparison_results = self.comparator.compare_default_vs_optimized(results, algorithm_name, comparison_save_path)
        else:
            # 分析所有算法
            comparison_save_path = f"{plots_dir}default_vs_all_algorithms.png" if save_plots else None
            comparison_results = self.comparator.compare_all_algorithms_vs_default(results, comparison_save_path)
        
        print("对比分析完成")
        return comparison_results
    
    def run_full_analysis(self, algorithms=None, max_iterations=GENERATIONS, 
                         save_results=True, save_plots=False, 
                         results_filename=None, plots_dir='./plots/'):
        """
        运行完整分析（一键运行所有功能）
        
        参数:
            algorithms: 要比较的算法列表
            max_iterations: 最大迭代次数
            save_results: 是否保存结果
            save_plots: 是否保存图片
            results_filename: 结果文件名
            plots_dir: 图片保存目录
            
        返回:
            results: 完整分析结果
        """
        print(f"\n{'='*80}")
        print("开始克里金插值参数优化完整分析")
        print(f"{'='*80}")
        
        # 1. 运行算法比较
        results = self.run_algorithm_comparison(algorithms, max_iterations, save_results, results_filename)
        
        # 2. 可视化结果
        self.visualize_results(results, save_plots, plots_dir)
        
        # 3. 默认参数对比分析
        comparison_results = self.analyze_vs_default(results, save_plots=save_plots, plots_dir=plots_dir)
        
        # 4. 生成最终报告
        self._generate_final_report(results, comparison_results)
        
        print(f"\n{'='*80}")
        print("完整分析完成")
        print(f"{'='*80}")
        
        return {
            'optimization_results': results,
            'comparison_results': comparison_results
        }
    
    def _print_comparison_summary(self, results):
        """
        打印比较总结
        """
        if not results:
            return
        
        print(f"\n{'='*80}")
        print("算法比较总结")
        print(f"{'='*80}")
        
        # 创建总结表
        summary_df = self.utils.create_results_summary(results)
        print(summary_df.to_string(index=False))
        
        # 找出最佳算法
        best_algorithm = summary_df.iloc[0]['算法']
        best_score = summary_df.iloc[0]['Best Score']
        print(f"\n最佳算法: {best_algorithm} (分数: {best_score:.6f})")
        
    def _generate_final_report(self, results, comparison_results):
        """
        生成最终报告
        """
        print(f"\n{'='*80}")
        print("最终分析报告")
        print(f"{'='*80}")
        
        if results:
            # 算法性能排名
            summary_df = self.utils.create_results_summary(results)
            print(f"\n算法性能排名:")
            for i, row in summary_df.iterrows():
                print(f"{row['排名']}. {row['算法']} - Score: {row['Best Score']:.6f}, "
                      f"执行时间: {self.utils.format_execution_time(row['执行时间(秒)'])}")
        
        if comparison_results:
            # 参数优化效果
            print(f"\n参数优化效果:")
            for alg_name, comp_result in comparison_results.items():
                if isinstance(comp_result, dict) and 'improvements' in comp_result:
                    improvements = comp_result['improvements']
                    print(f"{alg_name}: RMSE改进 {improvements['rmse_improvement']:+.2f}%, "
                          f"MAE改进 {improvements['mae_improvement']:+.2f}%, "
                          f"R²改进 {improvements['r2_improvement']:+.2f}%")


def main():
    """
    主函数 - 演示如何使用优化比较器
    """
    # 创建比较器
    comparator = KrigeOptimizationComparison()
    
    # 运行完整分析
    results = comparator.run_full_analysis(
        algorithms=['GA', 'PSO', 'Bayesian'],  # 可以选择特定算法
        max_iterations=100,  # 设置较小的迭代次数用于快速测试
        save_results=True,
        save_plots=True,
        plots_dir='./plots/'
    )
    
    return results


if __name__ == "__main__":
    # 运行主函数
    main()
