'''
Funtion: 主要用于分析原数据集，读取json文件，生成训练集和验证集、
Data: 2020.11.18
'''
import os
from os.path import join
import json
from pandas import  DataFrame
import pandas as pd
import random

'''
Funtion: 读取json文件，获取文件名和标签
Args:
    path:数据集所在的路径
    despath:保存生成的image_list所在的路径
Return:
    None
'''
def Source(path,despath):
    # os.walk 返回 （dirpath, dirnames, filenames）
    d={"image_name":[],"target":[]}
    for file_path in os.walk(path):
        print("数据集大小为：%d"%len(file_path[2]))
        for file in file_path[2]:
            try:
                with open(join(path,file)) as f:
                    json_file=json.load(f)
                    target = json_file["meta"]["clinical"]["benign_malignant"]
                    d["image_name"].append(file)
                    d['target'].append(target)
            except:
                pass
    DataFrame.from_dict(d).to_csv(join(despath,"image_list.csv"),index=False)

'''
Function: 获取数据集分布情况
Args: image_list
Return: None
'''
def Source_Length(file):
    d=pd.read_csv(file)
    benign,malignant=0,0    #统计各自的数目
    for i in d["target"]:
        if i=='benign':
            benign+=1
        else:
            malignant+=1
    print("数据大小为：%d, benign大小为：%d, malignant大小为：%d"%(len(d["target"]),benign,malignant))

'''
Function: 按照比例划分列表
Args: 
    full_list: 需要划分的列表
    shuffle: 是否打乱数据集
    ratio: 划分比例
'''
def split(full_list,shuffle=False,ratio=0.2):
    total=len(full_list)
    offset=int(total*ratio)
    if total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1=full_list[:offset]
    sublist_2=full_list[offset:]
    return sublist_1,sublist_2

'''
Function: 划分训练集和测试集：
Args: 
    image_list 数据集列表
    des_path 目标文件夹
    balance:是否平衡划分，True:正例：负例=1：1
Return:
    目标文件夹下生成train.txt和val.txt
'''
def dataset(image_list,des_path='./',shuffle=True,ratio=0.8,balance=False):
    d=pd.read_csv(image_list)
    benign,malignant=[],[]  #保存各自的文件名，先分成两类，再从中进行划分
    for i in range(len(d["image_name"])):
        if d["target"][i]=='benign':
            benign.append(d["image_name"][i])
        else:
            malignant.append(d["image_name"][i])
    print("begin数目:%d, malignant数目:%d"%(len(benign),len(malignant)))
    if balance:
        benign=benign[:len(malignant)]  #TODO：这里简单的取前几个元素，可以更改
        print("平衡数据集划分，正例:负例=%d:%d"%(len(benign),len(malignant)))
    train1, vol1 = split(benign, shuffle=shuffle, ratio=ratio)
    train2, vol2 = split(malignant, shuffle=shuffle, ratio=ratio)
    train,val={"image_name":[],"target":[]},{"image_name":[],"target":[]} #保存划分的结果，并且保存对于的标签
    for i in train1:
        train["image_name"].append(i)
        train["target"].append('benign')
    for i in train2:
        train["image_name"].append(i)
        train["target"].append('malignant')
    for i in vol1:
        val["image_name"].append(i)
        val["target"].append('benign')
    for i in vol2:
        val["image_name"].append(i)
        val["target"].append('malignant')
    DataFrame.from_dict(train).to_csv(join(des_path,"train.csv"),index=False)
    DataFrame.from_dict(val).to_csv(join(des_path, "val.csv"),index=False)
    print("训练集数目%d,验证集数目%d"%(len(train["target"]),len(val['target'])))

if __name__=='__main__':
    ##统计数据集，并且生成image_list.csv
    # path='../Data/Descriptions'
    # Source(path,despath='.')

    # #查看数据集的大小
    # file='./image_list.csv'
    # Source_Length(file)

    #划分测试集和验证集
    file = './image_list.csv'
    dataset(file,balance=True)
