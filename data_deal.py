import csv
from pandas import DataFrame
import numpy as np
import torch

#TODO:这个求法只是为了让程序暂且通过，后续需要更改：1.计算方式，2.函数实现方式
def get_dictory():
    features=[]
    with open('./data/0.csv') as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            temp = []
            for i in row[1:]:
                temp.append(list(map(float, i.replace('\n', '').replace('[', '').replace(']', '').split())))
            features.append(temp)
    dictory=np.zeros((8,20))
    for dim1 in range(len(features)):
        for dim2 in range(len(features[0])):
            df = DataFrame(features[dim1][dim2])
            dictory[dim1][dim2]=np.around(float(df.mean()),decimals=3)
    return dictory

def deal_Na(features):
    dictory=get_dictory()
    for dim1 in range(len(features)):
        for dim2 in range(len(features[0])):
            df = DataFrame(features[dim1][dim2])
            if float(df.mean()) == float(df.mean()):  # 缺失值==缺失值--->False
                df = df.fillna(value=df.mean())  # 如果均值不为0，就用平均值填充
            else:
                df = df.fillna(value=dictory[dim1][dim2])
            features[dim1][dim2] = torch.tensor(df.to_numpy().flatten())
    return features

if __name__=='__main__':
    pass