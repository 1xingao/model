class DrillData:
    def __init__(self, name="", location=[], layers=[], thickness=[]):
        self.drill_name = name  # 钻孔名称
        self.drill_location = location  # 钻孔的 xy 坐标
        self.drill_layers = layers  # 钻孔层的属性列表
        self.drill_layers_thickness = {}  # 每个属性对应的厚度

        for i, data in enumerate(thickness):
            self.drill_layers_thickness[layers[i]] = data

    def x(self):
        return self.drill_location[0]

    def y(self):
        return self.drill_location[1]



