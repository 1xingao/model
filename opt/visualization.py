# 克里金插值优化可视化模块
import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from scipy.spatial.distance import pdist, squareform

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 常量定义
DEFAULT_GRID_RESOLUTION = 100
FIGURE_SIZE = (20, 15)
VARIOGRAM_MODEL = 'spherical'


class KrigeVisualization:
    """
    克里金插值优化可视化类
    """
    
    def __init__(self):
        self.colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    def plot_semivariogram_comparison(self, results, save_path=None):
        """
        绘制半变异函数比较图
        
        参数:
            results: 算法结果字典
            save_path: 图片保存路径
        """
        if not results:
            print("请先运行算法比较")
            return
        
        # 使用第一个算法的数据
        first_alg = list(results.keys())[0]
        x_train, y_train, z_train = results[first_alg]['train_data']
        
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
        
        for i, (alg_name, result) in enumerate(results.items()):
            params = result['best_params']
            nugget = params['nugget']
            range_val = params['range']
            sill = params['sill']
            
            # 球状模型
            gamma_fit = self._spherical_model(h_fit, nugget, range_val, sill)
            
            plt.plot(h_fit, gamma_fit, color=self.colors[i % len(self.colors)], 
                    linewidth=2, label=f'{alg_name} 优化')
        
        plt.xlabel('距离 h')
        plt.ylabel('半方差 γ(h)')
        plt.title('不同优化算法的半变异函数比较')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interpolation_comparison(self, results, grid_resolution=DEFAULT_GRID_RESOLUTION, 
                                    save_path=None):
        """
        绘制插值结果比较图
        
        参数:
            results: 算法结果字典
            grid_resolution: 网格分辨率
            save_path: 图片保存路径
        """
        if not results:
            print("请先运行算法比较")
            return
        
        # 使用第一个算法的数据
        first_alg = list(results.keys())[0]
        x_train, y_train, z_train = results[first_alg]['train_data']
        x_test, y_test, z_test = results[first_alg]['test_data']
        
        # 创建插值网格
        x_all = np.concatenate([x_train, x_test])
        y_all = np.concatenate([y_train, y_test])
        
        grid_x = np.linspace(x_all.min(), x_all.max(), grid_resolution)
        grid_y = np.linspace(y_all.min(), y_all.max(), grid_resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        
        # 计算每个算法的插值结果
        n_algs = len(results)
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
        for i, (alg_name, result) in enumerate(results.items()):
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
        for j in range(len(results) + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_results_summary(self, results, save_path=None):
        """
        绘制结果总结图
        
        参数:
            results: 算法结果字典
            save_path: 图片保存路径
        """
        if not results:
            print("请先运行算法比较")
            return
        
        alg_names = list(results.keys())
        scores = [results[alg]['best_score'] for alg in alg_names]
        runtimes = [results[alg]['runtime'] for alg in alg_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 适应度比较
        bars1 = ax1.bar(alg_names, scores, color=self.colors[:len(alg_names)])
        ax1.set_ylabel('RMSE')
        ax1.set_title('算法性能比较')
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱状图上显示数值
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}', ha='center', va='bottom')
        
        # 运行时间比较
        bars2 = ax2.bar(alg_names, runtimes, color=self.colors[:len(alg_names)])
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
    
    def plot_semivariogram_single(self, x, y, z, params, label, ax):
        """
        绘制单个半变异函数
        
        参数:
            x, y, z: 坐标和属性值
            params: 变差函数参数
            label: 标签
            ax: 绘图轴
        """
        # 计算实验半方差
        coords = np.vstack((x, y)).T
        dists = squareform(pdist(coords))
        semivariances = 0.5 * (z[:, None] - z[None, :]) ** 2
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
        
        # 绘制实验半方差
        if label == '默认参数':
            ax.scatter(bin_centers, gamma_means, alpha=0.7, label=f'{label} 实验点')
        
        # 绘制理论变差函数
        h_fit = np.linspace(0, h.max(), 200)
        gamma_fit = self._spherical_model(h_fit, params[0], params[1], params[2])
        ax.plot(h_fit, gamma_fit, linewidth=2, label=f'{label} 理论曲线')
        ax.set_xlabel('距离 h')
        ax.set_ylabel('半方差 γ(h)')
    
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
