from torch.utils.data import Dataset
import csv
from os.path import join
import numpy as np
import torch
from torchvision import transforms
from data_deal import deal_all
from pandas import DataFrame

class dataloader(Dataset):
    def __init__(self,path,transforms=None,data_set='train',class_type="B"):
        self.path=path
        self.transforms=transforms
        self.data_set=data_set+'.txt'
        self.class_type=class_type
        #self.class_d={"NC":1,"MCI":2,'AD':3}        #TODO:这里需要支持三分类
        self.class_d = {"NC": 0, 'AD': 1} if self.class_type=='B' else {"NC": 0, "MCI": 1, 'AD': 2}
        self.features=[]
        self.labels=[]

        with open(self.data_set) as f:
            for line in f.readlines():
                line=line.strip().split()
                if self.class_type=='B' and line[1]=='MCI':
                    continue
                self.labels.append(self.class_d[line[1]])
                self.features.append(line[0])
        #这里初始化就将处理好的数据加载进内存中来
        self.data=deal_all(self.features,self.path,self.transforms)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        label = self.labels[item]
        label = torch.tensor(label)
        features=self.data[item]
        return features,label
if __name__=='__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    d=dataloader(path='./data',transforms=transform)
    f,l=d[0]
    print(f.shape)
    print(f[6][0],l)