from collections import defaultdict

def build_covering_order(boreholes):
    """
    构建所有钻孔共同的上下地层关系，并推导最复杂的层序
    """
    pair_counts = defaultdict(int)
    all_layers = set()

    # 统计每个地层对的上下顺序
    for layers in boreholes.values():
        all_layers.update(layers)
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                pair_counts[(layers[i], layers[j])] += 1

    # 拓扑排序建图
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    for (upper, lower), count in pair_counts.items():
        successors[upper].add(lower)
        predecessors[lower].add(upper)

    # Kahn拓扑排序算法
    result = []
    candidates = [l for l in all_layers if not predecessors[l]]
    while candidates:
        current = sorted(candidates)[0]  # 稳定排序
        result.append(current)
        candidates.remove(current)
        for succ in successors[current]:
            predecessors[succ].remove(current)
            if not predecessors[succ]:
                candidates.append(succ)

    if len(result) < len(all_layers):
        raise Exception("存在循环依赖，无法推导出稳定的层序")

    return result

def fill_borehole_to_full_sequence(borehole_layers, full_sequence):
    """
    根据最复杂层序补齐某钻孔，缺失部分以 "0_地层名" 填补
    """
    final = []
    existing = set(borehole_layers)
    for layer in full_sequence:
        if layer in borehole_layers:
            final.append(layer)
        else:
            final.append(f"0_{layer}")
    return final

def apply_to_all_boreholes(boreholes):
    full_sequence = build_covering_order(boreholes)
    transformed = {
        k: fill_borehole_to_full_sequence(v, full_sequence)
        for k, v in boreholes.items()
    }
    return full_sequence, transformed

# 示例数据
boreholes = {
    "B1": ["黄土","细粒砂岩","泥岩","中粒砂岩","砂质泥岩","泥岩","细粒砂岩"],
    "B2": ["黄土","砂质泥岩"],
    "B3": ["黄土","砂质泥岩","细粒砂岩","砂质泥岩","泥岩","细粒砂岩"],
    "B4": ["黄土","砾石","砂质泥岩","中粒砂岩"],
    "B5": ["黄土","泥岩","细粒砂岩"]
}


full_sequence, standardized = apply_to_all_boreholes(boreholes)

print("最复杂地层层序：", full_sequence)
for k, v in standardized.items():
    print(f"{k}: {v}")
