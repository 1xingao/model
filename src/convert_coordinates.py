import pandas as pd
from pyproj import Transformer

def convert_coordinates(input_path, output_path, to_crs):
    """
    将地层坐标文件中的经纬度坐标转换为指定的目标坐标系（WGS84 或 CGCS2000）。

    参数：
        input_path: 输入文件路径（地层坐标.xlsx）。
        output_path: 输出文件路径。
        to_crs: 目标坐标系（如 'EPSG:4326' 表示 WGS84，'EPSG:4490' 表示 CGCS2000）。
    """

    df = pd.read_excel(input_path)

    # 初始化坐标转换器（假设原始坐标系为 WGS84）
    transformer = Transformer.from_crs("EPSG:4326", to_crs, always_xy=True)

    def transform_row(row):
        x, y = transformer.transform(row['x'], row['y'])
        return pd.Series({'x': x, 'y': y})

    df[['x', 'y']] = df.apply(transform_row, axis=1)


    df.to_excel(output_path, index=False)
    print(f"转换完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    input_file = "./data/地层坐标.xlsx"
    output_file_wgs84 = "./data/地层坐标_WGS84.xlsx"
    output_file_cgcs2000 = "./data/地层坐标_CGCS2000.xlsx"

    # 转换为 WGS84
    convert_coordinates(input_file, output_file_wgs84, to_crs="EPSG:4326")

    # 转换为 CGCS2000
    convert_coordinates(input_file, output_file_cgcs2000, to_crs="EPSG:4490")
