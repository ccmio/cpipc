from prediction_model import *
import pandas as pd
import define
import matplotlib.pyplot as plt

target = [1710.6173348173872, 293.4367152585847, 391.2434431252802, 62.163478280008874, 0.007428968665785962,
          367.17496398158676, 55.51232048655796, 34.50186626462881, 195.33506593709296, 26.357874584352313,
          110.50913142225549, 416.9086696719244, 6.943509993569871, 15.344946788297127, 15.345111500546148,
          68.46219487437725, 348.0678174331914, 290.55149471725804, -10740.18808167453, 248.0, 89.4, 55.9, 20.6, 23.5,
          50.11, 727.8, 2.53, 8.57, 1.3, 6.69]
base = [4696.6507, 0.6894531, 0.6679688, 0.26625, 0.338789, 6.946113, 6.879375, -0.000351563, 44.95573075, 86.06048,
        137.9926, 421.1138325, 4.423842575, 30.730875, 30.7383865, 82.38895425, 261.885185, 0, 2695.48795, 248.0, 89.4,
        55.90, 20.60, 23.50, 50.11, 727.80, 2.53, 8.57, 1.30, 6.69]

df = pd.read_excel('data/data_total.xlsx')

obj133 = df[132:133][:]
ron_record = []
s_record = []
other0 = obj133.columns[0]
other1 = obj133.columns[1]
print(obj133)
count = 0
for idx in range(19):
    delta = define.delta[idx]
    print('调整中的特征：', idx, '调整次数：', count)
    if base[idx] < target[idx]:
        diff = target[idx] - base[idx]
        r = round(diff / delta) + 1
        print(obj133.iloc[0, idx + 2])
        ron_record.append(PredModel.predFunc(obj133.drop(other1, axis=1), 1))
        s_record.append(PredModel.predFunc(obj133.drop(other0, axis=1), 0))
        count += 1
        while r - 1:
            obj133.iloc[0, idx + 2] += delta
            print(obj133.iloc[0, idx + 2])
            ron_record.append(PredModel.predFunc(obj133.drop(other1, axis=1), 1))
            s_record.append(PredModel.predFunc(obj133.drop(other0, axis=1), 0))
            count += 1
            r -= 1
        obj133.iloc[0, idx + 2] = target[idx]
        print(obj133.iloc[0, idx + 2])
        ron_record.append(PredModel.predFunc(obj133.drop(other1, axis=1), 1))
        s_record.append(PredModel.predFunc(obj133.drop(other0, axis=1), 0))
        count += 1
    elif base[idx] > target[idx]:
        diff = base[idx] - target[idx]
        r = round(diff / delta) + 1
        print(obj133.iloc[0, idx + 2])
        ron_record.append(PredModel.predFunc(obj133.drop(other1, axis=1), 1))
        s_record.append(PredModel.predFunc(obj133.drop(other0, axis=1), 0))
        count += 1
        while r - 1:
            obj133.iloc[0, idx + 2] -= delta
            print(obj133.iloc[0, idx + 2])
            ron_record.append(PredModel.predFunc(obj133.drop(other1, axis=1), 1))
            s_record.append(PredModel.predFunc(obj133.drop(other0, axis=1), 0))
            r -= 1
            count += 1
        obj133.iloc[0, idx + 2] = target[idx]
        print(obj133.iloc[0, idx + 2])
        ron_record.append(PredModel.predFunc(obj133.drop(other1, axis=1), 1))
        s_record.append(PredModel.predFunc(obj133.drop(other0, axis=1), 0))
        count += 1
    else:
        continue
print(ron_record)
print(s_record)
plt.grid(ls='--')
plt.xlabel("操作调整次数", fontsize=14)
plt.ylabel("RON损失", fontsize=14)
plt.plot(ron_record)
plt.show()

plt.grid(ls='--')
plt.xlabel("操作调整次数", fontsize=14)
plt.ylabel("硫含量", fontsize=14)
plt.plot(s_record)
plt.show()
