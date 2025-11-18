import csv
from os.path import join
import os
from pandas import DataFrame
from pandas import Series
import pandas as pd
from matplotlib import pyplot as plt
import random
import math
from collections import defaultdict

#用来划分测试集和验证集
class data_statistics:
    def __init__(self,path,filename):
        self.path=path
        self.filename=filename
        #统计患病人数，正常人数
        self.dict={}
        # 源数据集中各自占比
        #self.origin = {'NC': 0, 'MCI': 0, 'AD': 0}
        self.origin = {'NC': {"count":0,"index":[]}, 'MCI': {"count":0,"index":[]}, 'AD': {"count":0,"index":[]}}
        # 统计年龄分布
        self.origin_age = {'NC': [], 'MCI': [], 'AD': []}
        self.origin_sex = {'NC': [], 'MCI': [], 'AD': []}
    #加载数据
    #def data_load(self):
        with open(join(self.path,self.filename)) as f:
            next(f)
            reader=csv.reader(f)
            for row in reader:
                self.dict[row[0]]=row[1:]
    #统计各个类别的数目
    #def data_ratio(self):
        for i,num in enumerate(self.dict['train_diagnose']):
            if float(num)==1:
                self.origin['NC']['count']+=1
                self.origin['NC']['index'].append(i)
                self.origin_age['NC'].append(round(float(self.dict['age'][i]),2))
                self.origin_sex['NC'].append(self.dict['sex'][i])
            elif float(num)==2:
                self.origin['MCI']['count'] += 1
                self.origin['MCI']['index'].append(i)
                self.origin_age['MCI'].append(round(float(self.dict['age'][i])))
                self.origin_sex['MCI'].append(self.dict['sex'][i])
            elif float(num)==3:
                self.origin['AD']['count'] += 1
                self.origin['AD']['index'].append(i)
                self.origin_age['AD'].append(round(float(self.dict['age'][i])))
                self.origin_sex['AD'].append(self.dict['sex'][i])

            #保存每个类别age和sex出现的频次
            self.count_age={}
            self.count_sex={}
            for i in self.origin_sex.keys():
                #计算频次
                sex=Series(self.origin_sex[i],dtype='float64').value_counts()
                self.count_sex[i]=sex
            for i in self.origin_age.keys():
                #计算频次
                #age=Series(origin_age[i]).value_counts()
                #计算平均值
                age = Series(self.origin_age[i],dtype='float64').mean()
                self.count_age[i]=round(age,2)
    #返回各个类别的数目
    def get_number_class(self):
        return self.origin
    #返回各个类别年龄的均值
    def get_age(self):
        return self.count_age
    #返回各个类别性别的频次
    def get_sex(self):
        return self.count_sex
    def hist_distributed_age_sex(self):
        # 绘制直方图
        plt.hist(self.origin_age['AD'], label='AD', alpha=0.4)
        plt.hist(self.origin_age['NC'], label='NC', alpha=0.4)
        plt.hist(self.origin_age['MCI'], label='MCI', alpha=0.4)
        plt.legend()
        plt.show()
    #ratio=[0.8,0.2]
    def divide_data(self,ratio=[0.8,0.2]):

        data_set={'train':{'NC': [], 'MCI': [], 'AD': []},'val':{'NC': [], 'MCI': [], 'AD': []}}
        for i in self.origin.keys():
            random.shuffle(self.origin[i]['index'])
            data_set['train'][i]=(self.origin[i]['index'][0:math.ceil(ratio[0]*self.origin[i]['count'])])
            data_set['val'][i]=(self.origin[i]['index'][math.ceil(ratio[0] * self.origin[i]['count']):])
        #删除原来的txt文件
        for name in os.listdir('./'):
            if name == 'requirements.txt':
                continue
            if name.endswith('.txt'):
                os.remove(join('./',name))
                print("Delete the file:"+join('./',name))

        #写入对应的txt文件
        for key in data_set.keys():
            with open(key+'.txt','a+') as f:
                for i in data_set[key].keys():
                    for j in data_set[key][i]:
                        f.write(str(j)+' '+i+'\n')  #TODO
        print("divide the data-set finished!")
        return data_set

if __name__=='__main__':
    d=data_statistics(path='./data',filename='data_result.csv')
    number=d.get_number_class()
    for i in number.keys():
        print(i,number[i])
    #划分数据集
    #divide=d.divide_data([0.8,0.2])
    # for key in divide.keys():
    #     for i in divide[key].keys():
    #         print(key,i,len(divide[key][i]))
    #age和sex
    print(d.get_age())
    print(d.get_sex())
    #求年龄平均值
    age=sum(d.get_age().values())/3
    print("averge_age:%f"%age)
    d.hist_distributed_age_sex()

