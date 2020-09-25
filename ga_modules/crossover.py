# 交叉算子
import numpy as np


class Cross:
    def __init__(self, buildings_new, cp):
        self.bn = buildings_new
        self.cp = cp

    # 单点交叉
    def crossover(self):
        children = []
        half = int(len(self.bn)/2)
        # 把种群对半分，一半父本一半母本
        father = self.bn[:half]
        mother = self.bn[half:]
        # 用shuffle打乱个体顺序
        np.random.shuffle(father)
        np.random.shuffle(mother)
        # 交配得到子代
        for i in range(half):
            if self.cp >= np.random.uniform(0, 1):
                key = np.random.randint(0, int(len(father[i])-11))
                son = father[i][: key] + (mother[i][key:])
                daughter = mother[i][: key] + (father[i][key:])
            else:
                son = father[i]
                daughter = mother[i]
            children.append(son)
            children.append(daughter)
        # 返回子代
        return children
