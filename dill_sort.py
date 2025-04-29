
import pandas as pd
from collections import Counter
from copy import deepcopy

# data
boreholes = {
    "BH01": ["填土", "黄土", "粉土", "粉砂", "砂岩"],
    "BH02": ["填土", "黄土", "粉砂", "砂岩", "泥岩"],
    "BH03": ["填土", "粉土", "黄土", "砂岩", "砾石"],#倒置
    "BH04": ["填土", "黄土", "粉土", "泥岩", "页岩"],#倒置
    "BH05": ["填土", "黄土", "粉砂", "页岩", "黏土"],#倒置
    "BH07": ["填土", "砂岩", "粉砂", "泥岩", "砾石"],#倒置
    "BH08": ["填土", "粉土", "黄土", "粉砂", "泥岩"],#倒置
    "BH09": ["填土", "砂岩", "砾石", "页岩", "粉土"],#倒置
    "BH10": ["填土", "粉砂", "黄土", "粉砂", "砂岩"],
    # "B1":["1","2","3","4","5"],
    # "B2":["1","2","3","2","4","5"],
    # "B3":["1","2","1","3","4","5"]
}

boreholes_thickness = {
    "BH01": [11,13,14,16,20],
    "BH02": [30,24,25,13,31],
    "BH03": [23,14,15,26,26],
    "BH04": [13,14,15,15,16],
    "BH05": [34,25,25,26,16],
    "BH07": [13,14,15,15,26],
    "BH08": [12,13,14,15,15],
    "BH09": [22,21,21,21,21],
    "BH10": [24,24,24,24,24],
    # "B1":["1","2","3","4","5"],
    # "B2":["1","2","3","2","4","5"],
    # "B3":["1","2","1","3","4","5"]
}

def get_unified_sequence(boreholes):
    # 1. 提取所有地层类型
    all_layers = set(layer for layers in boreholes.values() for layer in layers)
    all_layers = sorted(list(all_layers))  # 保持可读性排序
    layer_index = {layer: i for i, layer in enumerate(all_layers)}

    # 2. 初始化关系矩阵：delta[i][j] 累加
    C = [[0 for _ in all_layers] for _ in all_layers]

    # 3. 统计所有钻孔中地层的相对关系
    for bh_layers in boreholes.values():
        for i in range(len(bh_layers)):
            for j in range(len(bh_layers)):
                if i == j:
                    continue
                li, lj = bh_layers[i], bh_layers[j]
                idx_i, idx_j = layer_index[li], layer_index[lj]
                # 如果 i 在 j 上
                if i < j:
                    C[idx_i][idx_j] += 1
                else:
                    C[idx_i][idx_j] -= 1

    # 4. 计算每一层的总优先级（上方越多，得分越高）
    priority = {layer: 0 for layer in all_layers}
    for i, li in enumerate(all_layers):
        for j, lj in enumerate(all_layers):
            priority[li] += C[i][j]

    # 5. 排序得出统一层序（得分从高到低）
    sorted_layers = sorted(priority.items(), key=lambda x: -x[1])
    unified_sequence = [layer for layer, _ in sorted_layers]

    print("统一地层层序：")
    print(unified_sequence)
    return unified_sequence

# 为补全缺失地层的钻孔创建最大兼容层实现对重复层的0地层填充
def make_max_compatible_boreholes(boreholes: dict, unified_sequence: list):

    # 所有钻孔标准地层之间存在的异常层
    error_list = []
    borehole_keys = list(boreholes.keys())
    boreholes_matrix = list(boreholes.values())
    
    for i in range(len(unified_sequence)-1):
        error_layer = set()
        
        for idx,borehole in enumerate(boreholes_matrix):
            start = i
            while start < len(borehole) and Standard_names(borehole[start]) != unified_sequence[i]:
                start += 1
            end = start +1
            while end <len(borehole) and Standard_names(borehole[end]) != unified_sequence[i+1]:
                end += 1
            
            if end -start != 1:
                for j in range(start+1,end):
                    if(j >= len(borehole)): break
                    error_layer.add(borehole[j])
                    
        error_list.append(list(error_layer) if len(error_layer) > 0 else [])

    end_error_list = []
   
    for borehole in boreholes_matrix:
        index = len(borehole)-1
        while Standard_names(borehole[index]) != unified_sequence[-1]:
            index -= 1
        if index + 1 < len(borehole):
            end_error_list.extend(borehole[index+1:])
        
    
    error_list.append(list(set(end_error_list)))
    

    for i in range(len(error_list)):
        for j in range(len(error_list[i])):
            error_list[i][j] +=  "-"
    print(f"所有钻孔标准地层之间存在的异常层：")
    print(error_list)


    # 将所有的异常层以0厚度层的方式插入到没有该异常层的钻孔中
    res = {}
    for idx,borehole in enumerate(boreholes_matrix):
        unified_sequence_index = 0
        temp = []
        curr_layer_index = 0
        while curr_layer_index < len(borehole):
            if (unified_sequence_index >= len(unified_sequence)):
                temp.append(borehole[curr_layer_index])
                curr_layer_index += 1
                continue
            if(Standard_names(borehole[curr_layer_index]) == unified_sequence[unified_sequence_index]):
                temp.append(borehole[curr_layer_index])
                curr_layer_index += 1
                for error_layer in error_list[unified_sequence_index]:
                    if(curr_layer_index < len(borehole) and borehole[curr_layer_index] in error_layer):
                        temp.append(borehole[curr_layer_index])
                        curr_layer_index += 1
                    else:
                        temp.append(error_layer)
            unified_sequence_index += 1

        res[borehole_keys[idx]] = temp
    return res


# pandas标准化输出
def print_padas(standardized_boreholes):
    max_len = max(len(layers) for layers in standardized_boreholes.values())
# 对每个钻孔序列补齐到相同长度
    for bh in standardized_boreholes:
        length_diff = max_len - len(standardized_boreholes[bh])
        standardized_boreholes[bh] += ['一一'] * length_diff  
    df = pd.DataFrame(standardized_boreholes)
    print(df)


def Standard_names(layer_name):
    if "-"  in layer_name or "0" in layer_name:
        return layer_name[:-1]
    return layer_name

# 补全缺失层，并且将地层倒置的情况转化为地层重复（倒置地层上面插入0厚度层）
def complete_missing_layers(boreholes,unified_sequence):
    standardized_boreholes = {}
    problematic_boreholes = []

    for bh_name, bh_layers in boreholes.items():
        fixed_layers = []
        seen_layers = set()
        current_index = 0

        for layer in bh_layers:
            # 按统一层序推进，直到遇到当前 layer
            while current_index < len(unified_sequence) and unified_sequence[current_index] != layer and current_index < unified_sequence.index(layer):
                
                missing = unified_sequence[current_index]
                fixed_layers.append(missing+"0")  # 补插缺失层
                seen_layers.add(missing+"0")
                current_index += 1

            # 如果找到了匹配
            if current_index < len(unified_sequence) and unified_sequence[current_index] == layer:
                fixed_layers.append(layer)
                seen_layers.add(layer)
                current_index += 1
            else:
                # 如果地层不在统一层序中，直接添加
                fixed_layers.append(layer)

        # 后续层序补全（虚拟层）
        for i in range(current_index, len(unified_sequence)):
            fixed_layers.append(f"{unified_sequence[i]}0")

        # 检查是否重复
        
        if len(fixed_layers)!=len(unified_sequence):
            problematic_boreholes.append((bh_name, "地层重复"))

        standardized_boreholes[bh_name] = fixed_layers


    # 输出结果
    print("\n标准化后的钻孔层序（倒置转重复后）：")

    print_padas(deepcopy(standardized_boreholes))

    if problematic_boreholes:
        print("\n以下钻孔存在问题（已转化为地层重复）：")
        for bh, err in problematic_boreholes:
            print(f"{bh} - {err}")
    else:
        print("\n所有钻孔层序正常")


    return standardized_boreholes


def zero_thickness_filling(boreholes:dict,borehole_thickness:dict):

    res = {}

    for key,value in boreholes.items():
        thickness_layer = []
        layer_index = 0
        for i in range(len(value)):
            if "0" in value[i] or "-" in value[i]:
                thickness_layer.append(0)
            else:
                thickness_layer.append(borehole_thickness[key][layer_index])
                layer_index += 1
        res[key] = thickness_layer
    return res



def main():
    # 标准地层序
    unified_sequence = get_unified_sequence(boreholes)
    #去除缺失和倒置的钻孔数据
    standardized_boreholes = complete_missing_layers(boreholes,unified_sequence)
    #预处理完成的钻孔数据
    preconditioning_completes_drilling = make_max_compatible_boreholes(standardized_boreholes,unified_sequence)

    print_padas(preconditioning_completes_drilling)

    boreholes_thickness_filling = zero_thickness_filling(preconditioning_completes_drilling,boreholes_thickness)

    df = pd.DataFrame(boreholes_thickness_filling)
    print(df)

if __name__ == "__main__":
    main()
    



                


