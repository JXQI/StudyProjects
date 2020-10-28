import csv
from os.path import join
from pandas import DataFrame
from pandas import Series
import pandas as pd
from matplotlib import pyplot as plt

#用来划分测试集和验证集
path='./data'
dict={}
with open(join(path,'data_result.csv')) as f:
    next(f)
    reader=csv.reader(f)
    for row in reader:
        dict[row[0]]=row[1:]
#源数据集中各自占比
origin={'NC':0,'MCI':0,'AD':0}
#统计年龄分布
origin_age={'NC':[],'MCI':[],'AD':[]}
origin_sex={'NC':[],'MCI':[],'AD':[]}
for i,num in enumerate(dict['train_diagnose']):
    if float(num)==1:
        origin['NC']+=1
        origin_age['NC'].append(round(float(dict['age'][i]),2))
        origin_sex['NC'].append(dict['sex'][i])
    elif float(num)==2:
        origin['MCI'] += 1
        origin_age['MCI'].append(round(float(dict['age'][i])))
        origin_sex['MCI'].append(dict['sex'][i])
    elif float(num)==3:
        origin['AD'] += 1
        origin_age['AD'].append(round(float(dict['age'][i])))
        origin_sex['AD'].append(dict['sex'][i])
print(origin)
count_age={}
count_sex={}
for i in origin_sex.keys():
    sex=Series(origin_sex[i]).value_counts()
    count_sex[i]=sex
for i in origin_age.keys():
    #age=Series(origin_age[i]).value_counts()
    age = Series(origin_age[i]).mean()
    count_age[i]=round(age,2)
#求每个类别的平均年龄
print(count_age)
print(count_sex)
#绘制直方图
plt.hist(origin_age['AD'],label='AD',alpha=0.4)
plt.hist(origin_age['NC'],label='NC',alpha=0.4)
plt.hist(origin_age['MCI'],label='MCI',alpha=0.4)
plt.legend()
plt.show()
#必须有每个列表长度相同才可以用
# df_age=DataFrame(origin_age)
# df_sex=DataFrame(origin_sex)
# df_age(pd.value_counts())
# df_sex(pd.value_counts())


