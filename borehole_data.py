class boreholeData:
    def __init__(self, name="", location=[], layers=[], thickness=[]):
        self.borehole_name = name  # 钻孔名称
        self.borehole_location = location  # 钻孔的 xy 坐标
        self.borehole_layers = layers  # 钻孔层的属性列表
        self.borehole_layers_thickness = {}  # 每个属性对应的厚度

        for i, data in enumerate(thickness):
            self.borehole_layers_thickness[layers[i]] = data

    def x(self):
        return self.borehole_location[0]

    def y(self):
        return self.borehole_location[1]



