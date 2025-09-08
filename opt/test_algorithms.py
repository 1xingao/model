# 克里金插值参数优化算法测试脚本
import numpy as np
import pandas as pd
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

from genetic_algorithm_optimizer import GAKrigeOptimizer
from particle_swarm_optimizer import PSOKrigeOptimizer
from ant_colony_optimizer import ACOKrigeOptimizer
from differential_evolution_optimizer import DEKrigeOptimizer

def create_test_data():
    """
    创建测试数据
    """
    # 创建模拟的地层数据
    np.random.seed(42)
    n_points = 50
    
    x = np.random.uniform(0, 1000, n_points)
    y = np.random.uniform(0, 1000, n_points)
    z = 100 + 0.1 * x + 0.05 * y + np.random.normal(0, 5, n_points)
    
    # 创建DataFrame
    df = pd.DataFrame({
        '地层名称': ['松散层'] * n_points,
        'x': x,
        'y': y,
        'z': z
    })
    
    # 保存为Excel文件
    test_file = 'test_data.xlsx'
    df.to_excel(test_file, index=False)
    print(f"测试数据已保存到: {test_file}")
    
    return test_file

def test_algorithm(optimizer_class, name, **kwargs):
    """
    测试单个算法
    """
    print(f"\n{'='*40}")
    print(f"测试 {name} 算法")
    print(f"{'='*40}")
    
    try:
        # 创建优化器
        optimizer = optimizer_class(**kwargs)
        
        # 加载数据
        test_file = 'test_data.xlsx'
        train_data, test_data = optimizer.load_and_split_data(
            test_file, '松散层', 0.7, 42
        )
        
        # 定义参数边界
        x_train, y_train, z_train = train_data
        bounds = optimizer.define_parameter_bounds(x_train, y_train, z_train)
        
        print(f"参数边界: {bounds}")
        
        # 运行优化（短时间测试）
        best_score, best_params = optimizer.optimize(
            train_data, test_data, bounds, 'rmse', verbose=False
        )
        
        print(f"最优适应度: {best_score:.6f}")
        print(f"最优参数: {best_params}")
        print(f"{name} 测试成功 ✓")
        
        return True
        
    except Exception as e:
        print(f"{name} 测试失败 ✗")
        print(f"错误: {e}")
        return False

def main():
    """
    主测试函数
    """
    print("克里金插值参数优化算法测试")
    print("="*50)
    
    # 创建测试数据
    test_file = create_test_data()
    
    # 测试算法配置（使用较小的参数以快速测试）
    algorithms = [
        (GAKrigeOptimizer, "遗传算法", {"generations": 20, "population_size": 10}),
        (PSOKrigeOptimizer, "粒子群算法", {"iterations": 20, "n_particles": 10}),
        (ACOKrigeOptimizer, "蚁群算法", {"iterations": 20, "n_ants": 20}),
        (DEKrigeOptimizer, "差分进化算法", {"generations": 20, "population_size": 10}),
    ]
    
    # 测试结果
    results = []
    
    for optimizer_class, name, kwargs in algorithms:
        success = test_algorithm(optimizer_class, name, **kwargs)
        results.append((name, success))
    
    # 总结测试结果
    print(f"\n{'='*50}")
    print("测试结果总结")
    print(f"{'='*50}")
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name:<15}: {status}")
    
    successful_tests = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print(f"\n总计: {successful_tests}/{total_tests} 个算法测试通过")
    
    # 清理测试文件
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\n已清理测试文件: {test_file}")

if __name__ == '__main__':
    main()
