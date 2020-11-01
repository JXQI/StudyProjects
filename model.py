import torch
import torch.nn as nn
import torch.nn.functional as F
from loader import dataloader
from torchvision import transforms
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#两层FCN
class Linear_2(nn.Module):
    def __init__(self,isDrop=True,p=0.2):
        super(Linear_2, self).__init__()
        self.isDrop=[isDrop,p]
        self.name='Linear_2'
        if self.isDrop:
            self.features=nn.Sequential(
                nn.Linear(7*20*100,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(self.isDrop[1]),
                nn.Linear(4096, 2),
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(in_features=7 * 20 * 100, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=2),
            )
    def forward(self,x):
        x=x.view(-1,7*20*100)
        x=self.features(x)

        return x
#三层FCN
class Linear_3(nn.Module):
    def __init__(self,isDrop=True,p=0.2):
        super(Linear_3, self).__init__()
        self.isDrop=[isDrop,p]
        self.name='Linear_3'
        if self.isDrop:
            self.features=nn.Sequential(
                nn.Linear(7*20*100,8192),
                nn.ReLU(inplace=True),
                nn.Dropout(self.isDrop[1]),
                nn.Linear(8192, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 2),
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(7 * 20 * 100, 8192),
                nn.ReLU(inplace=True),
                nn.Linear(8192, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 2),
            )
    def forward(self,x):
        x=x.view(-1,7*20*100)
        x=self.features(x)

        return x

#模型调用窗口
class Model:
    def __init__(self,Weight_path=' ',net='Linear_2',pretrained=False,isDrop=(False,0.2)):
        self.net=net
        self.pretrained=pretrained
        self.Weight_path=Weight_path
        self.isDrop=isDrop

    def Net(self):
        if self.net=='Linear_2':
            Model=Linear_2(isDrop=self.isDrop)
        elif self.net=='Linear_3':
            Model = Linear_3(isDrop=self.isDrop)
        if self.pretrained:
            Model.load_state_dict(torch.load(self.Weight_path))
        return Model

if __name__=='__main__':
    model=Model(Weight_path='./Weights/best_Linear_0_55.pth')
    transform = transforms.Compose([transforms.ToTensor()])
    d = dataloader(path='./data', transforms=transform)
    feature,label=d[0]
    #处理缺失的值有两种情况：1.单个值缺失(平均值代替) 2.整组缺失(特定的组替代)
    dictory=np.array([[0]*20,[1]*20,[2]*20,[3]*20,[4]*20,[5]*20,[6]*20,[7]*20])     #TODO:求出一个均值来代替
    for dim1 in range(len(feature)):
        for dim2 in range(len(feature[0])):
            df=DataFrame(feature[dim1][dim2])
            if float(df.mean())==float(df.mean()):        #缺失值==缺失值--->False
                df=df.fillna(value=df.mean())  #如果均值不为0，就用平均值填充
            else:
                df=df.fillna(value=dictory[dim1][dim2])
            feature[dim1][dim2]=torch.tensor(df.to_numpy().flatten())
    #测试
    y=model.Net()
    print(y.name)
    print(y.features)
