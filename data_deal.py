import csv
from pandas import DataFrame,Series
import numpy as np
import torch
from os.path import join


def normalize(data):
    #data.shape (7,20,100)
    for dim1 in range(len(data)):#7
        for dim2 in range(len(data[dim1])):#20
            #dim1,dim2=7,0
            df = Series(data[dim1][dim2])
            #print(df.max(),df.min())
            df_norm=(df-df.mean())/(df.std())
            data[dim1][dim2]=df_norm.values
            #print(df_norm.max(),df_norm.min())
            break
        break
    return data
#TODO:这个求法只是为了让程序暂且通过，后续需要更改：1.计算方式，2.函数实现方式
def get_dictory():
    features=[]
    with open('./data/1.csv') as f:
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
            df_norm=(df-df.mean())/(df.std())
            dictory[dim1][dim2]=np.around(float(df_norm.mean()),decimals=3)
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

def deal_all(self_features,self_path,self_transforms):
    data=[]
    for item in range(len(self_features)):
        print(item,len(self_features))
        filename = join(self_path, self_features[item] + '.csv')
        features = []  # 暂时保存每一个病人的特征（20，100，7）
        with open(filename) as f:
            next(f)
            reader = csv.reader(f)
            for row in reader:
                temp = []
                for i in row[1:]:
                    temp.append(list(map(float, i.replace('\n', '').replace('[', '').replace(']', '').split())))
                features.append(temp)
        # TODO:这里暂时丢掉最后一维特征，为了保证数据数量级一样，避免特征消失features[:7]
        #features = np.around(np.array(features[:7], dtype=np.float32), decimals=3)  # TODO:为啥后边有那么多的0
        features = np.around(np.array(features, dtype=np.float32), decimals=3)
        features=normalize(features)
        features = features.transpose((1, 2, 0))  # TODO:需用补充一下转换的时候各种转置关系，以及和torch的转换关系

        # 转化成tensor类型
        if self_transforms:
            features = self_transforms(features)  # TODO:处理数据缺失的问题
        # 在此处暂时加入数据处理部分：    #TODO:后续需要统一，而且写成可调用的函数或者类
        features = deal_Na(features)
        data.append(features)

    return data

if __name__=='__main__':
    get_dictory()