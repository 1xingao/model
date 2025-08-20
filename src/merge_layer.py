# merge_boreholes.py
# -*- coding: utf-8 -*-
"""
将地层表中每个钻孔按规则合并，保持横向 4 列为一组（钻孔名称/地层名称/深度/厚度）的版式。

规则：
1) 将 中粒砂岩、细粒砂岩、粗粒砂岩、粉砂岩 合并为 “砂岩”。
2) 将以“泥岩”结尾的统一为 “泥岩”。
3) 将以“土”结尾的统一为 “土层”。
4) 将 细沙、风积沙 以及名称以“砂/沙”结尾（含中砂/细砂/粗砂/砂层 等）统一为 “砂层”。
5) 煤层仅保留：煤3-1、煤4-2、煤5-2、煤5-3；其他煤层并入相邻层（默认并入上一层；若位于最顶层，则并入下一层）。
6) 同一钻孔内相邻同类（归一化后同名）地层合并，厚度与底深（“深度”为底深）累加更新。

用法：
    python merge_boreholes.py --input 地层统计.xlsx --sheet Sheet1 --output 地层统计_合并结果.xlsx
"""
import argparse
import re
from typing import List, Dict, Any

import numpy as np
import pandas as pd


KEEP_COAL = {"煤3-1", "煤4-2", "煤5-2", "煤5-3"}


def base(col: str) -> str:
    """返回列的基础名（去掉 .1 / .2 等后缀）。"""
    return col.split(".")[0]


def normalize_name(name: str) -> str:
    """根据规则进行地层名称归一化。"""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    if s == "" or s.lower() == "nan":
        return ""

    # 优先：以“土”结尾 → 土层（例如：黄土、红土、细沙土等）
    if s.endswith("土"):
        return "土层"

    # 砂岩族合并
    if s in {"中粒砂岩", "细粒砂岩", "粗粒砂岩", "粉砂岩"}:
        return "砂岩"

    # 泥岩族合并
    if s.endswith("泥岩"):
        return "泥岩"

    # 砂层族（细沙、风积沙、以及以“砂/沙”结尾的散体；含中砂/细砂/粗砂/砂层）
    if s in {"细沙", "风积沙"} or re.search(r"(中|细|粗)?砂$", s) or s.endswith("沙") or s == "砂层":
        return "砂层"

    return s


def coal_key(name: str) -> str:
    """标准化煤层名字以匹配是否应保留（去掉可能的“层”字）。"""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    if s.endswith("层"):
        s = s[:-1]
    return s


def split_groups(columns: List[str]) -> List[List[str]]:
    """将列按每 4 列（钻孔名称/地层名称/深度/厚度）分组。"""
    groups = [columns[i : i + 4] for i in range(0, len(columns), 4)]
    # 基本格式校验
    for g in groups:
        if len(g) != 4 or [base(c) for c in g] != ["钻孔名称", "地层名称", "深度", "厚度"]:
            raise ValueError(f"列分组不符合预期：{g}")
    return groups


def process_one_borehole(sub_df: pd.DataFrame) -> pd.DataFrame:
    """处理单个钻孔（四列子表）。返回合并后的四列 DataFrame。"""
    sub = sub_df.copy()
    sub.columns = ["钻孔名称", "地层名称", "深度", "厚度"]

    # 清洗空行
    sub = sub.dropna(how="all", subset=["地层名称", "深度", "厚度"])
    for c in ["深度", "厚度"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=["地层名称", "深度", "厚度"])
    if sub.empty:
        return pd.DataFrame(columns=["钻孔名称", "地层名称", "深度", "厚度"])

    # 计算顶深（假设“深度”为底深，自地表 0 向下；厚度 = 底深 - 顶深）
    sub["顶深"] = sub["深度"] - sub["厚度"]

    # 按顶深自上而下排序
    sub = sub.sort_values("顶深", kind="mergesort").reset_index(drop=True)

    # 先把不保留的煤层并入相邻层
    layers = sub.to_dict("records")
    cleaned = []
    i = 0
    while i < len(layers):
        L = layers[i]
        raw = str(L["地层名称"]).strip()
        is_coal = raw.startswith("煤")
        k = coal_key(raw)

        if is_coal and k not in KEEP_COAL:
            th = float(L["厚度"]) if pd.notna(L["厚度"]) else 0.0
            if cleaned:
                # 并入上一层：上一层底深与厚度增加
                cleaned[-1]["深度"] = float(cleaned[-1]["深度"]) + th
                cleaned[-1]["厚度"] = float(cleaned[-1]["厚度"]) + th
            else:
                # 若位于最顶层则并入下一层（若存在）：抬升下一层的顶深
                if i + 1 < len(layers):
                    nxt = layers[i + 1]
                    new_top = float(L["顶深"])
                    nxt["顶深"] = new_top
                    nxt["厚度"] = float(nxt["深度"]) - float(nxt["顶深"])
                    layers[i + 1] = nxt
                # 否则丢弃该孤立煤层
            i += 1
            continue
        else:
            cleaned.append(L)
            i += 1

    if not cleaned:
        return pd.DataFrame(columns=["钻孔名称", "地层名称", "深度", "厚度"])

    # 归一化名称 + 相邻同类层合并
    merged_rows = []
    for rec in cleaned:
        raw = str(rec["地层名称"]).strip()
        k = coal_key(raw)
        if raw.startswith("煤") and k in KEEP_COAL:
            final_name = k  # 统一为不带“层”的写法
        else:
            final_name = normalize_name(raw)

        if final_name == "":
            continue

        top = float(rec["顶深"])
        bottom = float(rec["深度"])
        thick = float(rec["厚度"])

        if not merged_rows:
            merged_rows.append(
                {
                    "钻孔名称": rec["钻孔名称"],
                    "地层名称": final_name,
                    "顶深": top,
                    "深度": bottom,
                    "厚度": thick,
                }
            )
        else:
            prev = merged_rows[-1]
            if prev["地层名称"] == final_name:
                # 相邻同类 → 合并
                prev["深度"] = max(prev["深度"], bottom)
                prev["厚度"] = prev["深度"] - prev["顶深"]
                merged_rows[-1] = prev
            else:
                merged_rows.append(
                    {
                        "钻孔名称": rec["钻孔名称"],
                        "地层名称": final_name,
                        "顶深": top,
                        "深度": bottom,
                        "厚度": thick,
                    }
                )

    out = pd.DataFrame(merged_rows, columns=["钻孔名称", "地层名称", "深度", "厚度", "顶深"])
    # 输出时不暴露“顶深”列
    out = out[["钻孔名称", "地层名称", "深度", "厚度"]]
    return out


def merge_workbook(input_path: str, sheet_name: str, output_path: str) -> pd.DataFrame:
    """处理整个工作表并保存到 output_path，返回最终 DataFrame。"""
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    groups = split_groups(list(df.columns))

    processed_per_bh = []
    max_len = 0
    # 逐钻孔处理
    for g in groups:
        processed = process_one_borehole(df[g])
        processed_per_bh.append(processed)
        max_len = max(max_len, len(processed))

    # 水平拼接并对齐行数
    final_blocks = []
    for idx, processed in enumerate(processed_per_bh):
        pad = max_len - len(processed)
        if pad > 0:
            processed = pd.concat(
                [
                    processed,
                    pd.DataFrame(
                        [{"钻孔名称": np.nan, "地层名称": np.nan, "深度": np.nan, "厚度": np.nan}]
                        * pad
                    ),
                ],
                ignore_index=True,
            )
        # 添加列后缀避免重名
        suffix = "" if idx == 0 else f".{idx}"
        processed = processed.rename(columns={c: (c if suffix == "" else f"{c}{suffix}") for c in processed.columns})
        final_blocks.append(processed.reset_index(drop=True))

    final_df = pd.concat(final_blocks, axis=1)
    # 保存
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, index=False, sheet_name="合并结果")
    return final_df


def main():
    # parser = argparse.ArgumentParser(description="钻孔地层数据横向合并与归一化处理")
    # parser.add_argument("--input", required=True, help="输入 Excel 文件路径（含工作表 Sheet1 的四列表格）")
    # parser.add_argument("--sheet", default="Sheet1", help="工作表名称，默认 Sheet1")
    # parser.add_argument("--output", required=True, help="输出 Excel 文件路径")
    # args = parser.parse_args()
    input_path = "./data/地层统计.xlsx"
    sheet_name = "Sheet1"
    output_path = "./data/合并结果.xlsx"

    final_df = merge_workbook(input_path, sheet_name, output_path)
    print(f"已生成：{output_path}，共 {len(final_df)} 行；列数 {len(final_df.columns)}。")


if __name__ == "__main__":
    main()
