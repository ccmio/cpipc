# 选择算子
import random
import math


class Select:
    def __init__(self, buildings, value, s_value):
        self.buildings = buildings
        self.value = value
        self.s_value = s_value

    def selection(self):
        min_p = min(self.value)
        max_p = max(self.value)
        min_idx = self.value.index(min_p)
        max_idx = self.value.index(max_p)
        remain_p = self.buildings[min_idx]
        self.buildings[max_idx] = remain_p  # 强制剔除最差个体增加最强个体的基因浓度

        min_s = min(self.s_value)
        max_s = max(self.s_value)
        min_s_idx = self.s_value.index(min_s)
        max_s_idx = self.s_value.index(max_s)
        if max_s_idx != max_idx: # 保证ron损失减少的情况下尽量多留存低硫基因
            remain_s = self.buildings[min_s_idx]
            self.buildings[max_s_idx] = remain_p

        fitness_sum = []
        buildings_new = []
        sumx = 0
        # softmax函数处理每个个体的权值，可扩大化差异并获得较理想的收敛结果
        for values in self.value:
            sumx += math.exp(values)
        # 轮盘赌算法模拟优胜劣汰
        for i in range(len(self.value)):
            if i == 0:
                fitness_sum.append(math.exp(self.value[i])/sumx)
            else:
                fitness_sum.append(math.exp(self.value[i])/sumx + fitness_sum[i-1])

        for i in range(len(self.value)):
            rand = random.uniform(0, 1)
            for j in range(len(self.value)):
                if j == 0:
                    if 0 < rand <= fitness_sum[j]:
                        buildings_new.append(self.buildings[j])
                else:
                    if fitness_sum[j-1] < rand <= fitness_sum[j]:
                        buildings_new.append(self.buildings[j])

        # 返回本代幸存个体
        return buildings_new





