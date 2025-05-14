import pandas as pd
import os

def wide_to_long(wide_df):
    # 所有钻孔列（跳过地层名称列）
    borehole_cols = wide_df.columns[1:]

    all_records = []

    for col in borehole_cols:
        borehole_name = col.split()[0]  # "BH01" from "BH01 (厚度)"
        for idx, row in wide_df.iterrows():
            layer = row["地层名称"]
            thickness = row[col]
            all_records.append({
                "钻孔编号": borehole_name,
                "地层": layer,
                "厚度": thickness
            })

    # 保持每个钻孔自身的顺序（逐列展开）
    long_df = pd.DataFrame(all_records)

    return long_df


def long_to_wide(long_df):
    # 保留原始地层顺序
    unique_layers = long_df["地层"].drop_duplicates()
    long_df["地层"] = pd.Categorical(long_df["地层"], categories=unique_layers, ordered=True)

    pivot_df = long_df.pivot_table(
        index="地层", 
        columns="钻孔编号", 
        values="厚度", 
        aggfunc="first"
    ).reset_index()

    pivot_df.columns = ["地层名称"] + [f"{col} (厚度)" for col in pivot_df.columns[1:]]
    return pivot_df

def convert_table(input_path, output_path):
    df = pd.read_excel(input_path)

    if "地层名称" in df.columns:
        print("✅ 检测为宽表格式，正在转换为长表...")
        result_df = wide_to_long(df)
    elif "钻孔编号" in df.columns and "地层" in df.columns and "厚度" in df.columns:
        print("✅ 检测为长表格式，正在转换为宽表...")
        result_df = long_to_wide(df)
    else:
        raise ValueError("❌ 表格格式无法识别，缺少必要字段。")

    result_df.to_excel(output_path, index=False)
    print(f"🎉 转换完成，已保存为: {output_path}")


if __name__ == "__main__":
    input_file = "borehole_data.xlsx"
    output_file = "转换后结果.xlsx"
    
    convert_table(input_file, output_file)
