# 适应度函数直接使用5个位置基因的加权总分
from prediction_model import *

S = 0
RON = 1


class Fit:
    def __init__(self, buildings, person):
        self.buildings = buildings
        self.person = person

    def fitness(self):
        value = []
        s_value = []
        other0 = self.person.columns[0]
        other1 = self.person.columns[1]
        # 设定不同位置基因的不同权重
        # 一号位基因即为self.buildings[i][0]：1代表A11，2代表A12....5代表A15，其余以此类推
        for obj in self.buildings:
            for j in range(0, len(obj)-11):
                self.person.iloc[0, j+2] = obj[j]
            value.append(PredModel.predFunc(self.person.drop(other1, axis=1), RON))
            s_value.append(PredModel.predFunc(self.person.drop(other0, axis=1), S))

        # 返回种群中每个个体的基因型权值和，即适应度
        return value, s_value
