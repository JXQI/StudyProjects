from torch.utils.data import Dataset
import csv
from os.path import join
import numpy as np
import torch
from torchvision import transforms

class dataloader(Dataset):
    def __init__(self,path,transforms=None,data_set='train'):
        self.path=path
        self.transforms=transforms
        self.data_set=data_set+'.txt'
        self.class_d={"NC":1,"MCI":2,'AD':3}
        self.features=[]
        self.labels=[]

        with open(self.data_set) as f:
            for line in f.readlines():
                line=line.strip().split()
                self.features.append(line[0])
                self.labels.append(self.class_d[line[1]])
    def __len__(self):
        return len(self.features)
    def __getitem__(self, item):
        filename=join(self.path,self.features[item]+'.csv')
        features=[]
        #print(filename)
        with open(filename) as f:
            next(f)
            reader=csv.reader(f)
            for row in reader:
                temp=[]
                for i in row[1:]:
                    temp.append(list(map(float,i.replace('\n','').replace('[','').replace(']','').split())))
                features.append(temp)
        features=np.around(np.array(features,dtype=np.float32),decimals=3)     #TODO:为啥后边有那么多的0
        label = self.labels[item]
        features=features.transpose((1,2,0))       #TODO:需用补充一下转换的时候各种转置关系，以及和torch的转换关系
        if self.transforms:
            features=self.transforms(features)      #TODO:处理数据缺失的问题
            label=torch.tensor(label)


        return features,label
if __name__=='__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    d=dataloader(path='./data',transforms=transform)
    f,l=d[550]
    print(len(d))
    print(f[0][0],l)