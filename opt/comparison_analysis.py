# 克里金插值优化对比分析模块
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from visualization import KrigeVisualization

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

VARIOGRAM_MODEL = 'spherical'


class KrigeComparison:
    """
    克里金插值参数默认vs优化对比分析类
    """
    
    def __init__(self):
        self.visualizer = KrigeVisualization()
    
    def compare_default_vs_optimized(self, results, algorithm_name=None, save_path=None):
        """
        对比默认参数与优化参数的效果
        
        参数:
            results: 算法结果字典
            algorithm_name: 指定算法名称，如果为None则使用第一个算法
            save_path: 图片保存路径
        """
        if not results:
            print("请先运行算法比较")
            return
        
        # 选择要对比的算法
        if algorithm_name is None:
            algorithm_name = list(results.keys())[0]
        elif algorithm_name not in results:
            print(f"算法 {algorithm_name} 不在结果中")
            return
        
        result = results[algorithm_name]
        x_train, y_train, z_train = result['train_data']
        x_test, y_test, z_test = result['test_data']
        optimized_params = result['best_params']
        
        print(f"\n{'='*60}")
        print(f"默认参数 vs {algorithm_name} 优化参数对比")
        print(f"{'='*60}")
        
        # 1. 创建默认和优化的克里金模型
        ok_default = OrdinaryKriging(
            x_train, y_train, z_train,
            variogram_model=VARIOGRAM_MODEL,
            verbose=False,
            enable_plotting=False
        )
        
        ok_optimized = OrdinaryKriging(
            x_train, y_train, z_train,
            variogram_model=VARIOGRAM_MODEL,
            variogram_parameters=[
                optimized_params['nugget'],
                optimized_params['range'],
                optimized_params['sill']
            ],
            verbose=False,
            enable_plotting=False
        )
        
        # 2. 获取默认参数
        default_params = ok_default.variogram_model_parameters
        
        # 3. 在测试集上进行预测
        z_pred_default, _ = ok_default.execute('points', x_test, y_test)
        z_pred_optimized, _ = ok_optimized.execute('points', x_test, y_test)
        
        # 4. 计算误差指标
        metrics = self._calculate_metrics(z_test, z_pred_default, z_pred_optimized)
        
        # 5. 打印对比结果
        self._print_comparison_results(default_params, optimized_params, metrics)
        
        # 6. 绘制对比图
        comparison_results = self._plot_detailed_comparison(
            z_test, z_pred_default, z_pred_optimized,
            default_params, optimized_params, metrics,
            x_train, y_train, z_train, algorithm_name, save_path
        )
        
        return comparison_results
    
    def compare_all_algorithms_vs_default(self, results, save_path=None):
        """
        对比所有算法与默认参数的效果
        
        参数:
            results: 算法结果字典
            save_path: 图片保存路径
        """
        if not results:
            print("请先运行算法比较")
            return
        
        comparison_summary = {}
        
        print(f"\n{'='*80}")
        print("所有算法 vs 默认参数性能对比总结")
        print(f"{'='*80}")
        
        for alg_name in results.keys():
            print(f"\n--- {alg_name} 算法 ---")
            comparison_results = self.compare_default_vs_optimized(results, alg_name, save_path=None)
            comparison_summary[alg_name] = comparison_results
        
        # 创建总结图表
        self._plot_summary_comparison(comparison_summary, save_path)
        
        return comparison_summary
    
    def _calculate_metrics(self, z_test, z_pred_default, z_pred_optimized):
        """
        计算性能指标
        """
        rmse_default = np.sqrt(np.mean((z_test - z_pred_default) ** 2))
        rmse_optimized = np.sqrt(np.mean((z_test - z_pred_optimized) ** 2))
        
        mae_default = np.mean(np.abs(z_test - z_pred_default))
        mae_optimized = np.mean(np.abs(z_test - z_pred_optimized))
        
        r2_default = 1 - np.sum((z_test - z_pred_default) ** 2) / np.sum((z_test - np.mean(z_test)) ** 2)
        r2_optimized = 1 - np.sum((z_test - z_pred_optimized) ** 2) / np.sum((z_test - np.mean(z_test)) ** 2)
        
        return {
            'rmse_default': rmse_default,
            'rmse_optimized': rmse_optimized,
            'mae_default': mae_default,
            'mae_optimized': mae_optimized,
            'r2_default': r2_default,
            'r2_optimized': r2_optimized
        }
    
    def _print_comparison_results(self, default_params, optimized_params, metrics):
        """
        打印对比结果
        """
        print("\n参数对比:")
        print(f"{'参数':<12} {'默认值':<15} {'优化值':<15} {'改进':<15}")
        print("-" * 60)
        print(f"{'Nugget':<12} {default_params[0]:<15.6f} {optimized_params['nugget']:<15.6f} {((optimized_params['nugget'] - default_params[0])/default_params[0]*100):+.2f}%")
        print(f"{'Range':<12} {default_params[1]:<15.6f} {optimized_params['range']:<15.6f} {((optimized_params['range'] - default_params[1])/default_params[1]*100):+.2f}%")
        print(f"{'Sill':<12} {default_params[2]:<15.6f} {optimized_params['sill']:<15.6f} {((optimized_params['sill'] - default_params[2])/default_params[2]*100):+.2f}%")
        
        print("\n性能对比:")
        print(f"{'指标':<12} {'默认参数':<15} {'优化参数':<15} {'改进':<15}")
        print("-" * 60)
        print(f"{'RMSE':<12} {metrics['rmse_default']:<15.6f} {metrics['rmse_optimized']:<15.6f} {((metrics['rmse_default'] - metrics['rmse_optimized'])/metrics['rmse_default']*100):+.2f}%")
        print(f"{'MAE':<12} {metrics['mae_default']:<15.6f} {metrics['mae_optimized']:<15.6f} {((metrics['mae_default'] - metrics['mae_optimized'])/metrics['mae_default']*100):+.2f}%")
        print(f"{'R²':<12} {metrics['r2_default']:<15.6f} {metrics['r2_optimized']:<15.6f} {((metrics['r2_optimized'] - metrics['r2_default'])/abs(metrics['r2_default'])*100):+.2f}%")
    
    def _plot_detailed_comparison(self, z_test, z_pred_default, z_pred_optimized,
                                default_params, optimized_params, metrics,
                                x_train, y_train, z_train, algorithm_name, save_path):
        """
        绘制详细对比图
        """
        fig = plt.figure(figsize=(20, 12))
        
        # 6.1 预测值 vs 真实值散点图
        ax1 = plt.subplot(2, 3, 1)
        plt.scatter(z_test, z_pred_default, alpha=0.6, label='默认参数')
        plt.scatter(z_test, z_pred_optimized, alpha=0.6, label=f'{algorithm_name}优化')
        plt.plot([z_test.min(), z_test.max()], [z_test.min(), z_test.max()], 'r--', label='理想线')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6.2 残差图
        ax2 = plt.subplot(2, 3, 2)
        residuals_default = z_test - z_pred_default
        residuals_optimized = z_test - z_pred_optimized
        plt.scatter(z_pred_default, residuals_default, alpha=0.6, label='默认参数')
        plt.scatter(z_pred_optimized, residuals_optimized, alpha=0.6, label=f'{algorithm_name}优化')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6.3 误差分布直方图
        ax3 = plt.subplot(2, 3, 3)
        plt.hist(residuals_default, bins=20, alpha=0.6, label='默认参数', density=True)
        plt.hist(residuals_optimized, bins=20, alpha=0.6, label=f'{algorithm_name}优化', density=True)
        plt.xlabel('残差')
        plt.ylabel('密度')
        plt.title('残差分布直方图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6.4 参数对比柱状图
        ax4 = plt.subplot(2, 3, 4)
        params_names = ['Nugget', 'Range', 'Sill']
        default_values = [default_params[0], default_params[1], default_params[2]]
        optimized_values = [optimized_params['nugget'], optimized_params['range'], optimized_params['sill']]
        
        x_pos = np.arange(len(params_names))
        width = 0.35
        plt.bar(x_pos - width/2, default_values, width, label='默认参数', alpha=0.8)
        plt.bar(x_pos + width/2, optimized_values, width, label=f'{algorithm_name}优化', alpha=0.8)
        plt.xlabel('参数')
        plt.ylabel('参数值')
        plt.title('参数对比')
        plt.xticks(x_pos, params_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6.5 性能指标对比
        ax5 = plt.subplot(2, 3, 5)
        metrics_names = ['RMSE', 'MAE', 'R²']
        default_metrics = [metrics['rmse_default'], metrics['mae_default'], metrics['r2_default']]
        optimized_metrics = [metrics['rmse_optimized'], metrics['mae_optimized'], metrics['r2_optimized']]
        
        x_pos = np.arange(len(metrics_names))
        plt.bar(x_pos - width/2, default_metrics, width, label='默认参数', alpha=0.8)
        plt.bar(x_pos + width/2, optimized_metrics, width, label=f'{algorithm_name}优化', alpha=0.8)
        plt.xlabel('指标')
        plt.ylabel('数值')
        plt.title('性能指标对比')
        plt.xticks(x_pos, metrics_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6.6 半变异函数对比
        ax6 = plt.subplot(2, 3, 6)
        self.visualizer.plot_semivariogram_single(x_train, y_train, z_train, default_params, '默认参数', ax6)
        self.visualizer.plot_semivariogram_single(x_train, y_train, z_train, 
                                      [optimized_params['nugget'], optimized_params['range'], optimized_params['sill']], 
                                      f'{algorithm_name}优化', ax6)
        plt.title('半变异函数对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 返回对比结果
        comparison_results = {
            'default_params': {
                'nugget': default_params[0],
                'range': default_params[1], 
                'sill': default_params[2]
            },
            'optimized_params': optimized_params,
            'default_metrics': {
                'rmse': metrics['rmse_default'],
                'mae': metrics['mae_default'],
                'r2': metrics['r2_default']
            },
            'optimized_metrics': {
                'rmse': metrics['rmse_optimized'],
                'mae': metrics['mae_optimized'],
                'r2': metrics['r2_optimized']
            },
            'improvements': {
                'rmse_improvement': (metrics['rmse_default'] - metrics['rmse_optimized']) / metrics['rmse_default'] * 100,
                'mae_improvement': (metrics['mae_default'] - metrics['mae_optimized']) / metrics['mae_default'] * 100,
                'r2_improvement': (metrics['r2_optimized'] - metrics['r2_default']) / abs(metrics['r2_default']) * 100
            }
        }
        
        return comparison_results
    
    def _plot_summary_comparison(self, comparison_summary, save_path):
        """
        绘制总结对比图
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        alg_names = list(comparison_summary.keys())
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # RMSE改进对比
        rmse_improvements = [comparison_summary[alg]['improvements']['rmse_improvement'] for alg in alg_names]
        bars1 = ax1.bar(alg_names, rmse_improvements, color=colors[:len(alg_names)])
        ax1.set_ylabel('RMSE改进 (%)')
        ax1.set_title('RMSE改进对比')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # MAE改进对比
        mae_improvements = [comparison_summary[alg]['improvements']['mae_improvement'] for alg in alg_names]
        bars2 = ax2.bar(alg_names, mae_improvements, color=colors[:len(alg_names)])
        ax2.set_ylabel('MAE改进 (%)')
        ax2.set_title('MAE改进对比')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # R²改进对比
        r2_improvements = [comparison_summary[alg]['improvements']['r2_improvement'] for alg in alg_names]
        bars3 = ax3.bar(alg_names, r2_improvements, color=colors[:len(alg_names)])
        ax3.set_ylabel('R²改进 (%)')
        ax3.set_title('R²改进对比')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 综合改进对比（平均值）
        avg_improvements = [(rmse_improvements[i] + mae_improvements[i] + r2_improvements[i])/3 for i in range(len(alg_names))]
        bars4 = ax4.bar(alg_names, avg_improvements, color=colors[:len(alg_names)])
        ax4.set_ylabel('平均改进 (%)')
        ax4.set_title('综合改进对比')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 在柱状图上显示数值
        for bars, values in [(bars1, rmse_improvements), (bars2, mae_improvements), 
                           (bars3, r2_improvements), (bars4, avg_improvements)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                bars[0].axes.text(bar.get_x() + bar.get_width()/2., height,
                                f'{value:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
