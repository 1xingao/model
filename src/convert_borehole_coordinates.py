
import pandas as pd
import numpy as np

def convert_to_local_coordinates(input_file, output_file=None):
    """
    将钻孔位置坐标转换为自建坐标系
    
    参数:
        input_file: 输入Excel文件路径
        output_file: 输出Excel文件路径 (可选)
    
    返回
        转换后的DataFrame和原点信息
    """
    
    print("="*50)
    print("钻孔坐标转换为自建坐标系")
    print("="*50)
    
    # 读取钻孔位置数据
    try:
        df = pd.read_excel(input_file)
        print(f"成功读取文件: {input_file}")
        print(f"数据行数: {len(df)}")
        print(f"列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None
    
    # 检查并标准化列名
    if 'x' not in df.columns or 'y' not in df.columns:
        print("错误: 文件中找不到x, y列")
        return None, None
    
    # 检查z列
    if 'z' not in df.columns:
        print("警告: 未找到z列，将设置为0")
        df['z'] = 0
    
    # 删除包含NaN的行
    original_count = len(df)
    df = df.dropna(subset=['x', 'y', 'z'])
    if len(df) < original_count:
        print(f"警告: 删除了{original_count - len(df)}行包含缺失值的数据")
    
    if len(df) == 0:
        print("错误: 没有有效的坐标数据")
        return None, None
    
    # 选择原点 (使用第一个点作为原点)
    origin_x = df['x'].iloc[0]
    origin_y = df['y'].iloc[0] 
    origin_z = df['z'].iloc[0]
    
    print(f"\n选定的坐标原点:")
    print(f"  原点X: {origin_x}")
    print(f"  原点Y: {origin_y}")
    print(f"  原点Z: {origin_z}")
    
    # 转换为自建坐标系 (相对于原点的偏移)
    df_local = df.copy()
    
    # 计算局部坐标 (单位保持不变)
    df_local['x_local'] = df['x'] - origin_x
    df_local['y_local'] = df['y'] - origin_y
    df_local['z_local'] = df['z'] - origin_z
    
    # 保留原始坐标作为参考
    df_local['x_original'] = df['x']
    df_local['y_original'] = df['y']
    df_local['z_original'] = df['z']
    
    # 更新主坐标列为局部坐标
    df_local['x'] = df_local['x_local']
    df_local['y'] = df_local['y_local']
    df_local['z'] = df_local['z_local']
    
    # 计算坐标范围
    x_range = df_local['x'].max() - df_local['x'].min()
    y_range = df_local['y'].max() - df_local['y'].min()
    z_range = df_local['z'].max() - df_local['z'].min()
    
    print(f"\n转换结果统计:")
    print(f"  X坐标范围: {df_local['x'].min():.2f} 至 {df_local['x'].max():.2f} (跨度: {x_range:.2f})")
    print(f"  Y坐标范围: {df_local['y'].min():.2f} 至 {df_local['y'].max():.2f} (跨度: {y_range:.2f})")
    print(f"  Z坐标范围: {df_local['z'].min():.2f} 至 {df_local['z'].max():.2f} (跨度: {z_range:.2f})")
    
    # 设置默认输出文件名
    if output_file is None:
        if input_file.endswith('.xlsx'):
            output_file = input_file.replace('.xlsx', '_局部坐标系.xlsx')
        elif input_file.endswith('.xls'):
            output_file = input_file.replace('.xls', '_局部坐标系.xls')
        else:
            output_file = input_file + '_局部坐标系.xlsx'
    
    # 保存结果
    try:
        df_local.to_excel(output_file, index=False)
        print(f"\n转换完成！")
        print(f"输出文件: {output_file}")
    except Exception as e:
        print(f"保存文件失败: {e}")
        return df_local, None
    
    # 返回原点信息
    origin_info = {
        'origin_x': origin_x,
        'origin_y': origin_y,
        'origin_z': origin_z,
        'x_range': x_range,
        'y_range': y_range,
        'z_range': z_range,
        'point_count': len(df_local)
    }
    
    return df_local, origin_info

def main():
    """主函数 - 使用示例"""
    
    # 输入文件路径 (请根据实际路径修改)
    input_file = "./data/real_data/钻孔位置统计.xlsx"
    
    # 执行转换
    df_result, origin_info = convert_to_local_coordinates(input_file)
    
    if df_result is not None and origin_info is not None:
        print("\n" + "="*50)
        print("转换成功！自建坐标系信息:")
        print("="*50)
        print(f"原点坐标: ({origin_info['origin_x']}, {origin_info['origin_y']}, {origin_info['origin_z']})")
        print(f"覆盖范围: {origin_info['x_range']:.2f} × {origin_info['y_range']:.2f}")
        print(f"钻孔数量: {origin_info['point_count']}")
        
        # 显示前几行数据作为预览
        print(f"\n转换后数据预览:")
        print(df_result[['钻孔名称', 'x', 'y', 'z', 'x_original', 'y_original']].head() if '钻孔名称' in df_result.columns else df_result[['x', 'y', 'z', 'x_original', 'y_original']].head())
    else:
        print("转换失败，请检查输入文件格式")

if __name__ == "__main__":
    main()