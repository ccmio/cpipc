# 变异算子，单点变异
import numpy as np
import random
import define


class Mutate:
    def __init__(self, children, mp):
        self.children = children
        self.mp = mp

    def mutation(self):
        for i in range(len(self.children)):
            if self.mp >= np.random.uniform(0, 1):
                key_n = np.random.randint(0, 2)
                while key_n:
                    key = np.random.randint(0, 19)
                # 一定概率下key位置的基因可随机变异
                    self.children[i][key] = random.uniform(define.limit[key][0], define.limit[key][1])
                    key_n -= 1
        # 返回变异后的子代
        return self.children
