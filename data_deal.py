import csv
from pandas import DataFrame,Series
import numpy as np
import os
from os.path import join
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from collections import defaultdict
import pandas as pd
import torch
from settings import AGE_MEAN

# #TODO:这个求法只是为了让程序暂且通过，后续需要更改：1.计算方式，2.函数实现方式
# def get_dictory():
#     features=[]  #NC代表
#     features_AD=[]
#     with open('./data/0.csv') as f:
#         next(f)
#         reader = csv.reader(f)
#         for row in reader:
#             temp = []
#             for i in row[1:]:
#                 temp.append(list(map(float, i.replace('\n', '').replace('[', '').replace(']', '').split())))
#             features.append(temp)
#     with open('./data/0.csv') as f:
#         next(f)
#         reader = csv.reader(f)
#         for row in reader:
#             temp = []
#             for i in row[1:]:
#                 temp.append(list(map(float, i.replace('\n', '').replace('[', '').replace(']', '').split())))
#             features_AD.append(temp)
#     dictory=np.zeros((8,20))
#     for dim1 in range(len(features)):
#         for dim2 in range(len(features[0])):
#             df = DataFrame(features[dim1][dim2])
#             df_AD=DataFrame(features_AD[dim1][dim2])
#             dictory[dim1][dim2]=np.around(float((df.mean()+df_AD.mean())/2),decimals=3)
#     return dictory

# def deal_all(self_features,self_path,self_transforms):
#     dictory_name="./dictory.csv"
#     path='./'
#     data = []
#     #判断字典是否存在，如果不存在就生成字典
#     if dictory_name not in os.listdir(path):
#         generate_dictory(path)
#     dictory = read_feature('./dictory.csv')
#     for item in range(len(self_features)):
#         print(item,len(self_features))
#         filename = join(self_path, self_features[item] + '.csv')
#         features = []  # 暂时保存每一个病人的特征（20，100，7）
#         with open(filename) as f:
#             next(f)
#             reader = csv.reader(f)
#             for row in reader:
#                 temp = []
#                 for i in row[1:]:
#                     temp.append(list(map(float, i.replace('\n', '').replace('[', '').replace(']', '').split())))
#                 features.append(temp)
#         # TODO:这里暂时丢掉最后一维特征，为了保证数据数量级一样，避免特征消失features[:7]
#         #features = np.around(np.array(features[:7], dtype=np.float32), decimals=3)  # TODO:为啥后边有那么多的0
#         features = np.around(np.array(features, dtype=np.float32), decimals=3)
#         features = deal_Na(features,dictory=dictory)
#         features=normalize(features)
#         features = features.transpose((1, 2, 0))  # TODO:需用补充一下转换的时候各种转置关系，以及和torch的转换关系
#         # 转化成tensor类型
#         if self_transforms:
#             features = self_transforms(features)
#         data.append(features)
#     return data

#用于读取单个的特征文件
def read_feature(csvfile):
    features=[]
    with open(csvfile) as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            temp = []
            for i in row[1:]:
                temp.append(list(map(float, i.replace('\n', '').replace('[', '').replace(']', '').split())))
            features.append(temp)
    features = np.around(np.array(features, dtype=np.float32), decimals=3)
    return features
#写入特征文件
def write_feature(feature,name):
    feature_keys = ['FA', 'MD', 'RD', 'AXD', 'liner', 'Curvature', 'torsion', 'volume']
    d_feature = defaultdict(list)
    for dim1 in range(len(feature)):#(8,20,100)
        for dim2 in range(len(feature[0])):
            d_feature[feature_keys[dim1]].append(feature[dim1][dim2])
    pd.DataFrame.from_dict(data=d_feature, orient='index').to_csv(name)

#TODO:这块处理方法：1.部分值缺失：字典查询对应位置的值+（本组均值-字典均值） 2.全部缺失：直接用字典的值替换
def deal_Na(features,dictory):
    for dim1 in range(len(features)):
        for dim2 in range(len(features[0])):
            df = DataFrame(features[dim1][dim2])
            #字典对应的位置
            df_dict = DataFrame(dictory[dim1][dim2])
            #所有值都为nan
            if all(np.isnan(features[dim1][dim2])):
                features[dim1][dim2]=dictory[dim1][dim2]
            else:
                for i,res in enumerate(np.isnan(features[dim1][dim2])):
                    if res:
                        features[dim1][dim2][i]=dictory[dim1][dim2][i]+(df.mean()-df_dict.mean())
    return features
#标准化数据
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
    return data
#融入年龄和性别信息,返回所有病人的年龄和性别信息
def age_sex():
    feature = []
    with open("./data/data_result.csv") as f:
        next(f)
        reader = csv.reader(f)
        for row in reader:
            temp = []
            for i in row[1:]:
                temp.append(list(map(float, i.replace('\n', '').replace('[', '').replace(']', '').split())))
            feature.append(temp)
    features=list(zip(feature[1], feature[2]))
    return features
#将预处理过的特征送进去loader.py,同时将处理后的数据存进新的文件中，避免重复计算
def load_save_data(self_features,self_path,self_transforms,Is_normalize):
    dictory_name = "dictory.csv"
    path = './'
    data ,filenames= load_all_features(self_features, self_path)
    if dictory_name not in os.listdir(path):
        generate_dictory(path='./result',d=data)    #产生数据分布图和查询字典
    dictory = read_feature(join(path,dictory_name))
    for item in range(len(data)):
        print("数据预处理进度：%d/%d"%(item,len(data)))
        features=data[item] #每一位病人的特征（8，20，100）
        features=deal_Na(features,dictory)
        if Is_normalize:features=normalize(features)
        write_feature(feature=features, name=join('data', "0_" + filenames[item] + '.csv'))
        features = features.transpose((1, 2, 0))  # TODO:需用补充一下转换的时候各种转置关系，以及和torch的转换关系
        if self_transforms:features = self_transforms(features)
        data[item]=features
    return data
#从处理后的文件中加载特征
def load_data(self_features,self_path,self_transforms):
    data = []
    age_sex_data = age_sex()  # 求的所有样本的年龄和性别信息
    for item in range(len(self_features)):
        print("数据加载进度：%d/%d" % (item, len(self_features)))
        filename = join(self_path, "0_"+self_features[item] + '.csv')
        features = []      #包含每个病人的纤维素特征
        all_feature = []  # 包含每个病人的纤维素特征和年龄性别特征
        age_sex_feature=age_sex_data[int(self_features[item])]
        #这里求（年龄值-平均值）/平均值 的结果作为特征输入
        age_sex_feature[1][0]=(age_sex_feature[1][0]-AGE_MEAN)/AGE_MEAN
        with open(filename) as f:
            next(f)
            reader = csv.reader(f)
            for row in reader:
                temp = []
                for i in row[1:]:
                    temp.append(list(map(float, i.replace('\n', '').replace('[', '').replace(']', '').split())))
                features.append(temp)
        # TODO:这里暂时丢掉最后一维特征，为了保证数据数量级一样，避免特征消失features[:7]
        # features = np.around(np.array(features[:7], dtype=np.float32), decimals=3)  # TODO:为啥后边有那么多的0
        features = np.around(np.array(features, dtype=np.float32), decimals=3)
        age_sex_feature=np.array(age_sex_feature,dtype=np.float32)
        features = features.transpose((1, 2, 0))  # TODO:需用补充一下转换的时候各种转置关系，以及和torch的转换关系
        if self_transforms:
            features = self_transforms(features)
            age_sex_feature=torch.tensor(age_sex_feature)

        all_feature.append(features)
        all_feature.append(age_sex_feature)
        data.append(all_feature)
    return data
#将./data下的所有特征文件读取到内存中来
def load_all_features(self_features,self_path):
    data = []
    for item in range(len(self_features)):
        print("数据加载进度：%d/%d"%(item,len(self_features)))
        filename = join(self_path, self_features[item] + '.csv')
        features = []
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
        data.append(features)
    return data,self_features
def generate_dictory(path,d):
    #data_list = [str(i) for i in range(700)]
    #d = load_all_features(data_list, "./data")
    # 这里用来存储求的的平均值
    feature_keys = ['FA', 'MD', 'RD', 'AXD', 'liner', 'Curvature', 'torsion', 'volume']
    d_feature = defaultdict(list)
    for dim1 in range(len(d[0])):#(8,20,100)
    #for dim1 in range(2):
        for dim2 in range(len(d[0][0])):
        #for dim2 in range(20):
            plt.figure(str(dim1) + '_' + str(dim2))
            ax = plt.gca()
            sum, count = np.zeros(100), 0
            for i in range(len(d)):
                data = np.array(d[i][dim1][dim2])
                if not (True in np.isnan(data)):
                    sum += data
                    count += 1
                    ax.scatter(range(len(data)), data, s=20, alpha=0.6)
            print(dim1, dim2, i, count)
            # 保存求得的平均值
            d_feature[feature_keys[dim1]].append(sum / count)
            plt.plot(range(len(sum)), sum / count, label="average")
            plt.savefig(join(path, str(dim1) + '_' + str(dim2) + '.jpg'))
            #TODO:这两行调试的时候可以看，但运行的时候最好注释掉，不然程序会卡住
            #plt.legend()
            #plt.show()
    # 保存dictory文件
    pd.DataFrame.from_dict(data=d_feature, orient='index').to_csv('dictory.csv')

if __name__=='__main__':
    #TODO:预处理数据，并且生成新的特征文件0_0.csv等
    self_features=[str(i) for i in range(700)]
    self_path='./data'
    self_transforms=transforms.Compose([transforms.ToTensor()])
    load_save_data(self_features, self_path, self_transforms, Is_normalize=True)
