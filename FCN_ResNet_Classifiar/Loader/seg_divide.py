'''
Funtion: 主要用于分析可以用于分割的原数据集，读取json文件，生成训练集和验证集、
Data: 2020.11.18
'''
import os
from os.path import join
import json
from pandas import  DataFrame
import pandas as pd
import random

'''
Function: 修改mask图像的名称，使其统一方便读取
Args: mask图像路径
Return: ISIC_0000000_expert.png 格式的名称
'''
def rename_mask(path):
    for files in os.walk(path):
        for i in files[2]:
            temp=i.split('_')
            if temp[2]!='expert.png':
                temp[2]='expert.png'
                os.rename(join(path,i),join(path,"_".join(temp)))

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
    d={"image_name":[],"target":[]} #target 这块可用可不用，需要后续再补充
    for file_path in os.walk(path):
        print("数据集大小为：%d"%len(file_path[2]))
        for file in file_path[2]:
            try:
                d["image_name"].append(file[:12])
                d['target'].append(0)
            except:
                pass
    DataFrame.from_dict(d).to_csv(join(despath,"segment_list.csv"),index=False)

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
def dataset(image_list,des_path='./',shuffle=True,ratio=0.8):
    d=pd.read_csv(image_list)
    print(d["image_name"])
    train_set, val_set = split(list(d["image_name"]), shuffle=shuffle, ratio=ratio)
    train,val={"image_name":train_set},{"image_name":val_set} #保存划分的结果，并且保存对于的标签
    DataFrame.from_dict(train).to_csv(join(des_path,"segment_train.csv"),index=False)
    DataFrame.from_dict(val).to_csv(join(des_path, "segment_val.csv"),index=False)
    print("训练集数目%d,验证集数目%d"%(len(train["image_name"]),len(val['image_name'])))

if __name__=='__main__':
    # 修改mask文件名
    path='../Data/data'
    rename_mask(path)

    # ##统计数据集，并且生成image_list.csv
    # path='../Data/data'
    # Source(path,despath='.')
    #
    # # #查看数据集的大小
    # file='./segment_list.csv'
    # Source_Length(file)
    #
    # #划分测试集和验证集
    # file = './segment_list.csv'
    # dataset(file)
