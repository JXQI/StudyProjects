from scipy.io import loadmat
import pandas as pd
import os
from os.path import  join
from collections import defaultdict
import numpy as np
#加载源shuju
from settings import URL,PATH,FILENAME
from download_from_url import download_dataset_from_url

#创建类用于将特征分开存储，方便后边对特征的提取已经融合方法实现
class data_load:
    def __init__(self,filename,save_path):
        self.file=loadmat(filename, mat_dtype=True)
        # 生成之前先删除之前文件中存在的
        self.path = save_path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

    # 脑内主要纤维束的特征
    # 提取每一个指标：这里将每个病人的特征单独存储一个文件，也就是说每个总共有700个文件，每个8*100
    def AFQ_feature(self):
        feature_keys = ['FA', 'MD', 'RD', 'AXD', 'liner', 'Curvature', 'torsion', 'volume']
        d_feature = defaultdict(list)
        print(self.file['train_set'].shape)
        for index in range(700):
            # 每个病人的特征分开保存
            for i in range(len(feature_keys)):
                # 八个特征分开
                for j in range(20):
                    # 20个纤维束
                    print(index, i, j)
                    # print(np.around(d['train_set'][j][0][i][index],decimals=3))
                    d_feature[feature_keys[i]].append(np.around(self.file['train_set'][j][0][i][index], decimals=3))
            pd.DataFrame.from_dict(data=d_feature, orient='index').to_csv(join(self.path, str(index) + '.csv'))
            # 需要注意这里添加进文件之后一定得将字典清空，不然会一直append到每个keys，导致最后一个文件累加进去了前边所有的值
            d_feature.clear()
        print("load and save the feature files have finished!!!")

    # 对年龄性别等指标归类，保存单个数据的特征，比如年龄性别等等
    def single_dimenon_feature(self):
        d_result = {}
        d_result_keys = ['train_diagnose', 'train_population', 'train_sites']
        for i in self.file.keys():
            if i in d_result_keys:
                if i == 'train_population':
                    d_result['sex'] = self.file[i][:, 0]
                    d_result['age'] = self.file[i][:, 1]
                else:
                    d_result[i] = self.file[i].flatten()
        pd.DataFrame.from_dict(data=d_result, orient='index').to_csv(join(self.path, 'data_result.csv'))
        print("load and save the dimenon_feature files have finished!!!")

if __name__=='__main__':
    filename = FILENAME
    download_dataset_from_url(url=URL,filename=FILENAME,path=PATH)
    save_path='./data'
    d=data_load(filename=filename,save_path=save_path)
    #如果已经存在，就不要反复执行，因为会重新生成文件
    d.single_dimenon_feature()
    #d.AFQ_feature()









