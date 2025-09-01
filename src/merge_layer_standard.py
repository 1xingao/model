# merge_boreholes_standardize.py
# -*- coding: utf-8 -*-
"""
按你给定的“标准地层”分段逻辑对每个钻孔进行归类与合并，并输出为横向四列一组
（钻孔名称/地层名称/深度/厚度）的表格。

分段与归类规则：
1) 顶部松散层：最上部连续出现的风积沙、黄土、红土等（以及以“土/沙”结尾的名称）
   统一并成一个“松散层”。
2) 从“松散层底”到“煤3-1顶”之间 → “含砾砂岩层”
3) 从“煤3-1顶”到“煤4-2顶”之间 → “砂岩泥岩混层”
4) 从“煤4-2顶”到“煤5-2顶”、以及“煤5-2顶”到“煤5-3顶”、
   以及“煤5-3”之后的所有层段 → “砂岩”
5) 标志煤层（煤3-1、煤4-2、煤5-2、煤5-3）始终独立保留，作为分段边界。
6) 最终对相邻同名层段进行连续合并。

★ 可选（按阈值保留“非标准地层”）：已在代码中以【注释】形式保留。
   如需启用，请按注释指示取消注释：
   - 当某原始地层名称既不是标志煤层、也不属于顶部松散层集合，
     且厚度 ≥ KEEP_NONSTANDARD_THRESHOLD 时，保留其原始名称，不并入标准类别。
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd



INPUT_PATH  = "./data/real_data/地层统计.xlsx"                  # 输入 Excel 文件
SHEET_NAME  = "Sheet1"                         # 工作表名
OUTPUT_PATH = "./data/real_data/地层统计_标准分段_合并结果.xlsx"   # 输出 Excel 文件
# 拼接模式: 'horizontal' 恢复原来按钻孔横向 4 列一组; 'vertical' 纵向堆叠
MERGE_MODE  = 'vertical'

# 【可选功能-按阈值保留“非标准地层”】：默认关闭
# 启用方法：
#  1) 取消下一行的注释，并设置一个合适的阈值（如 1.0 表示厚度≥1.0 的“非标准地层”保留原名）
#  2) 在 process_one_borehole 中标注的“阈值保留逻辑”代码块取消注释
# KEEP_NONSTANDARD_THRESHOLD = 1.0
# ============================================


# 标志煤层（独立保留 & 分段边界）
RESERVED_COALS = ("煤3-1", "煤4-2", "煤5-2", "煤5-3")

# 顶部“松散层”判定集合（仅用于“最上部连续段”的识别）
LOOSE_EXACT = {"风积沙", "黄土", "红土", "砂层", "细沙", "中砂", "粗砂", "粉土", "粘土","细沙土","风积沙土",
               "风积砂","粉砂","细砂"}


def base(col: str) -> str:
    """获取列的基础名（去掉 .1 / .2 等后缀）。"""
    return col.split(".")[0]


def split_groups(columns: List[str]) -> List[List[str]]:
    """按每 4 列分组为一个钻孔，并做结构校验。"""
    groups = [columns[i:i + 4] for i in range(0, len(columns), 4)]
    for g in groups:
        if len(g) != 4 or [base(c) for c in g] != ["钻孔名称", "地层名称", "深度", "厚度"]:
            raise ValueError(f"列分组不符合预期：{g}")
    return groups


def coal_key(name: str) -> str:
    """标准化煤层名（去掉可能的‘层’结尾）。"""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    return s[:-1] if s.endswith("层") else s


def is_loose_top_material(name: str) -> bool:
    """判断名称是否属于顶部松散介质（仅用于“最上部连续段”）。"""
    if not isinstance(name, str):
        return False
    s = name.strip()
    if s in LOOSE_EXACT:
        return True
    # 以“土/沙”结尾的名称（如：黄土、红土、粉土、粘土、细沙土、风积沙等）
    if s.endswith("土") or s.endswith("沙"):
        return True
    return False


def get_reserved_seam_tops(layers: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    获取保留煤层的“顶深”字典（若同名出现多次，取最上部那次的顶深）。
    返回示例：{"煤3-1": 30.2, "煤4-2": 58.0, ...}
    """
    tops: Dict[str, float] = {}
    for rec in layers:
        name = str(rec["地层名称"]).strip()
        key = coal_key(name)
        if key in RESERVED_COALS:
            top = float(rec["顶深"])
            if key not in tops or top < tops[key]:
                tops[key] = top
    return tops


def last_marker_above(seam_tops: Dict[str, float], layer_top: float) -> Optional[str]:
    """
    给定某层顶深，返回其“之上最近（最深）的标志煤层名”。
    若上方不存在任何标志煤层，返回 None。
    """
    marker = None
    depth_of_marker = -1.0
    for mk, mk_top in seam_tops.items():
        if mk_top <= layer_top and mk_top > depth_of_marker:
            marker = mk
            depth_of_marker = mk_top
    return marker


def assign_standard_label(marker_above: Optional[str]) -> str:
    """
    根据上方最近标志煤层决定标准类别：
      None        → “含砾砂岩层”      （松散层下至煤3-1之间）
      "煤3-1"     → “砂岩泥岩混层”    （煤3-1至煤4-2之间）
      其他（煤4-2/煤5-2/煤5-3） → “砂岩”
    """
    if marker_above is None:
        return "含砾砂岩层"
    if marker_above == "煤3-1":
        return "砂岩泥岩混层"
    return "砂岩"


def process_one_borehole(sub_df: pd.DataFrame) -> pd.DataFrame:
    """
    处理单个钻孔（四列子表），执行：
      - 顶部松散层合并
      - 标志煤层独立保留
      - 依据分段规则将其他层归入标准类别
      - 相邻同名层再连续合并
    返回四列 DataFrame（钻孔名称/地层名称/深度/厚度）
    """
    sub = sub_df.copy()
    sub.columns = ["钻孔名称", "地层名称", "深度", "厚度"]

    # 清洗与排序
    sub = sub.dropna(how="all", subset=["地层名称", "深度", "厚度"])
    for c in ["深度", "厚度"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=["地层名称", "深度", "厚度"])
    if sub.empty:
        return pd.DataFrame(columns=["钻孔名称", "地层名称", "深度", "厚度"])

    sub["顶深"] = sub["深度"] - sub["厚度"]
    sub = sub.sort_values("顶深", kind="mergesort").reset_index(drop=True)
    layers = sub.to_dict("records")

    final_rows: List[Dict[str, Any]] = []

    # —— A) 顶部松散层（只识别“最上部连续段”）——
    i = 0
    loose_stack: List[Dict[str, Any]] = []
    while i < len(layers) and is_loose_top_material(str(layers[i]["地层名称"]).strip()):
        loose_stack.append(layers[i])
        i += 1

    if loose_stack:
        top_top = float(loose_stack[0]["顶深"])
        bottom = float(loose_stack[-1]["深度"])
        final_rows.append({
            "钻孔名称": loose_stack[0]["钻孔名称"],
            "地层名称": "松散层",
            "顶深": top_top,
            "深度": bottom,
            "厚度": bottom - top_top
        })

    # —— B) 预计算标志煤层顶深 —— 
    seam_tops = get_reserved_seam_tops(layers)

    # —— C) 从顶部松散层之后开始逐层归类 —— 
    for j in range(i, len(layers)):
        rec = layers[j]
        raw = str(rec["地层名称"]).strip()
        key = coal_key(raw)
        top = float(rec["顶深"])
        bottom = float(rec["深度"])
        thick = float(rec["厚度"])

        # 标志煤层：独立保留
        if key in RESERVED_COALS:
            final_rows.append({
                "钻孔名称": rec["钻孔名称"],
                "地层名称": key,
                "顶深": top,
                "深度": bottom,
                "厚度": thick
            })
            continue

        # 标准类别：基于“上方最近的标志煤层”
        marker = last_marker_above(seam_tops, top)
        std_label = assign_standard_label(marker)

        # 默认：并入标准地层
        label = std_label

        # 【可选】阈值保留“非标准地层”的逻辑（默认关闭）————————————
        # 启用方法：
        #  1) 在文件顶部设置 KEEP_NONSTANDARD_THRESHOLD（取消注释并赋值）
        #  2) 取消下面 3 行注释
        #
        # if 'KEEP_NONSTANDARD_THRESHOLD' in globals():
        #     if (raw not in RESERVED_COALS) and (raw not in LOOSE_EXACT) and (thick >= KEEP_NONSTANDARD_THRESHOLD):
        #         label = raw  # 厚度达到阈值的“非标准地层”保留原始名称

        final_rows.append({
            "钻孔名称": rec["钻孔名称"],
            "地层名称": label,
            "顶深": top,
            "深度": bottom,
            "厚度": thick
        })

    # —— D) 相邻同名连续合并 —— 
    merged: List[Dict[str, Any]] = []
    for r in final_rows:
        if not merged:
            merged.append(r)
        else:
            prev = merged[-1]
            if prev["地层名称"] == r["地层名称"]:
                prev["深度"] = max(prev["深度"], r["深度"])
                prev["厚度"] = prev["深度"] - prev["顶深"]
                merged[-1] = prev
            else:
                merged.append(r)

    out = pd.DataFrame(merged, columns=["钻孔名称", "地层名称", "深度", "厚度", "顶深"])
    return out[["钻孔名称", "地层名称", "深度", "厚度"]]


def _merge_horizontal(df: pd.DataFrame, groups: List[List[str]]) -> pd.DataFrame:
    """原始横向拼接逻辑: 每个钻孔结果标准化后补齐行数并 4 列一组横向拼接。"""
    results: List[pd.DataFrame] = []
    max_rows = 0
    for g in groups:
        block = process_one_borehole(df[g])
        results.append(block)
        max_rows = max(max_rows, len(block))
    padded_blocks: List[pd.DataFrame] = []
    for idx, block in enumerate(results):
        pad = max_rows - len(block)
        if pad > 0:
            block = pd.concat([block, pd.DataFrame([
                {"钻孔名称": np.nan, "地层名称": np.nan, "深度": np.nan, "厚度": np.nan}
            ] * pad)], ignore_index=True)
        suffix = "" if idx == 0 else f".{idx}"
        block = block.rename(columns={c: (c if suffix == "" else f"{c}{suffix}") for c in block.columns})
        padded_blocks.append(block.reset_index(drop=True))
    return pd.concat(padded_blocks, axis=1)

def _merge_vertical(df: pd.DataFrame, groups: List[List[str]]) -> pd.DataFrame:
    """纵向堆叠版本: 所有钻孔记录放在四列表里。"""
    records: List[pd.DataFrame] = []
    for g in groups:
        block = process_one_borehole(df[g])
        if not block.empty:
            records.append(block)
    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame(columns=["钻孔名称","地层名称","深度","厚度"])

def add_layer_numbering(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每个钻孔的砂岩层添加编号，例如：砂岩 -> 砂岩_1, 砂岩_2。
    其他层保持不变。
    """
    df = df.copy()

    def update_layer_name(group):
        layer_counts = {}
        updated_layers = []
        for layer_name in group:
            if layer_name == "砂岩":
                if layer_name not in layer_counts:
                    layer_counts[layer_name] = 0
                layer_counts[layer_name] += 1
                updated_layers.append(f"{layer_name}_{layer_counts[layer_name]}")
            else:
                updated_layers.append(layer_name)
        return updated_layers

    df['地层名称'] = df.groupby('钻孔名称')['地层名称'].transform(update_layer_name)
    return df

def merge_workbook(input_path: str, sheet_name: str, output_path: str) -> pd.DataFrame:
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    groups = split_groups(list(df.columns))
    if MERGE_MODE == 'horizontal':
        final_df = _merge_horizontal(df, groups)
    else:
        final_df = _merge_vertical(df, groups)

    # 为每个钻孔的砂岩层添加编号
    final_df = add_layer_numbering(final_df)

    final_df.to_excel(output_path, index=False, sheet_name="合并结果")
    return final_df


# —— 直接在文件内调用（无需命令行）——
if __name__ == "__main__":
    final_df = merge_workbook(INPUT_PATH, SHEET_NAME, OUTPUT_PATH)
    mode_txt = '横向拼接(4列一组)' if MERGE_MODE=='horizontal' else '纵向堆叠'
    print(f"已生成：{OUTPUT_PATH}；行数={len(final_df)}，列数={len(final_df.columns)}，模式={mode_txt}。")
