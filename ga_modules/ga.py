# 主程序
# 因遗传算法特性，每次运行的进化图像不一致，收敛结果一致
# 可调整参数：初始种群个体数，基因交叉概率，基因变异概率，进化次数
# 需要自行安装的库：matplotlib, random
import matplotlib.pyplot as plt
import random
import define
import pandas as pd
from fitness import Fit
from selection import Select
from crossover import Cross
from mutation import Mutate
from prediction_model import *
import pandas as pd
import time
class GA:
    def __init__(self, path):
        self.df = pd.read_excel(path)
        self.result = []
        self.t = []
        self.record = []
        self.mutate_p = 0.3
        self.cross_p = 0.6

    def cul_single(self, person):
        other1 = person.columns[1]
        target_ron = PredModel.predFunc(person.drop(other1, axis=1), 1)
        true_target = person.iloc[0, 0]
        print('本样本真实ron损失：', true_target)
        print('预测模型认为的ron损失', target_ron)
        print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n')

        # 初始化随机种群
        buildings = []
        value_record = []
        s_record = []
        best_param = [i*0 for i in range(34)]
        percent_record = []
        # 20为种群内个体数
        for i in range(32):
            p = random.uniform(0.01, 0.99)
            temp = []
            for j in range(19):
                base_value = person.iloc[0, 2+j]
                right_value = define.limit[j][1] - person.iloc[0, 2+j]
                left_value = person.iloc[0, 2+j] - define.limit[j][0]
                if random.randint(0, 1):
                    value = base_value + p * right_value
                else:
                    value = base_value - p * left_value
                # value = random.uniform(define.limit[j][0], define.limit[j][1])
                temp.append(value)
            for j in range(19, 30):
                temp.append(person.iloc[0, j+2])
            buildings.append(temp)
        # 进化次数250次，建议不少于200次
        a = time.time()
        for i in range(1000):
            # 计算每个个体的适应度函数的值
            value, s_value = Fit(buildings, person).fitness()
            temp_min = min(value)
            value_record.append(temp_min)
            index = value.index(temp_min)
            temp_s = s_value[index]
            s_record.append(temp_s)
            reduce_percent = (target_ron - temp_min)/target_ron
            percent_record.append(reduce_percent)
            if reduce_percent > 0.301 and temp_s < 5:
            # if temp_min <= min(value_record) and temp_s < 5:
            #     best_param = [buildings[index], temp_min, reduce_percent, temp_s, i]
                best_param = buildings[index] + [temp_min, reduce_percent, temp_s, i]
                break
            print('本轮最小RON损失：', temp_min)
            print('RON损失降幅:{:.1f}%'.format(reduce_percent * 100))
            print('对应硫含量：', temp_s)
            print('================round', i, 'finished==================\n')
            # t用于存储最优个体的适应度，用以绘制最后的进化图像
            # record用于存储每次进化最优的个体，方便最后打印最优基因型
            # Select函数决定本代哪些个体可以活到下一代
            buildings_new = Select(buildings, value, s_value).selection()
            # 活下来的个体以0.7的概率进行交配产生子代，交配概率动态降低
            children = Cross(buildings_new, self.cross_p * max(1-reduce_percent/0.4, 0.1)).crossover()
            # 子代以0.3的基准概率基因突变，突变概率动态降低
            buildings = Mutate(children, self.mutate_p * max(1-reduce_percent/0.4, 0.1)).mutation()
        b = time.time()
        print(b-a)
        # print('\n最佳优化参数：', best_param[0], '\nRON损失：', best_param[1], '\n损失降幅：', best_param[2],
        #       '\n产品硫含量：', best_param[3], '\n所在轮次:', best_param[4])
        print(best_param)
        plt.title('RON损失值')
        plt.grid(ls='--')
        plt.xlabel("进化轮次", fontsize=14)
        plt.ylabel("RON损失", fontsize=14)
        plt.plot(value_record)
        # plt.show()

        plt.title('产品硫含量')
        plt.grid(ls='--')
        plt.xlabel("进化轮次", fontsize=14)
        plt.ylabel("硫含量", fontsize=14)
        plt.plot(s_record)
        # plt.show()

        plt.title('RON损失降低比率')
        plt.grid(ls='--')
        plt.xlabel("进化轮次", fontsize=14)
        plt.ylabel("降损比例", fontsize=14)
        plt.plot(percent_record)
        # plt.show()
        resultframe = pd.DataFrame([best_param])
        resultframe.to_csv('./result/opt_result.csv', mode='a', index=False, header=None)
        # return best_param

    def run(self):
        # 提取最优解的基因型
        for idx in range(95, self.df.shape[0]):
        # for idx in range(132, 134):
            row = self.df[idx:idx+1][:]
            self.result.append(self.cul_single(row))



