from scipy.io import loadmat
import pandas as pd
import os
from os.path import  join
from collections import defaultdict
import numpy as np


#加载数据集
d=loadmat('MCAD_AFQ_competition.mat', mat_dtype=True)
print(d.keys())

#生成之前先删除之前文件中存在的
path='./data'
if not os.path.isdir(path):
    os.mkdir(path)

#脑内主要纤维束的特征
#提取每一个指标：这里将每个病人的特征单独存储一个文件，也就是说每个总共有700个文件，每个8*100
feature_keys=['FA','MD','RD','AXD','liner','Curvature', 'torsion','volume']
d_feature=defaultdict(list)
print(d['train_set'].shape)
for index in range(700):
    #每个病人的特征分开保存
    for i in range(len(feature_keys)):
        #八个特征分开
        for j in range(20):
            #20个纤维束
            print(index,i,j)
            #print(np.around(d['train_set'][j][0][i][index],decimals=3))
            d_feature[feature_keys[i]].append(np.around(d['train_set'][j][0][i][index],decimals=3))
    pd.DataFrame.from_dict(data=d_feature,orient='index').to_csv(join(path,str(index)+'.csv'))
#对年龄性别等指标归类，保存单个数据的特征，比如年龄性别等等
d_result={}
d_result_keys=['train_diagnose','train_population','train_sites']
for i in d.keys():
    if i in d_result_keys:
        if i=='train_population':
            d_result['sex']=d[i][:,0]
            d_result['age'] =d[i][:,1]
        else:
            d_result[i] = d[i].flatten()
pd.DataFrame.from_dict(data=d_result,orient='index').to_csv(join(path,'data_result.csv'))

# print(d['fgnames'].shape)
# print(d['train_set'].shape)
# print(d['train_set'][0][0])
# print(len(d['train_set'][0][0]))
# print(d['train_set'][0][0][0].shape)