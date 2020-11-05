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

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.name='ConvNet'
        self.features1 = nn.Sequential(
            nn.Conv2d(100, 1, 1, stride=1),  #8*20
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(1, 20), stride=(1, 1))
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(100, 100,kernel_size=(1,20), stride=(1,1)),  # 8*20
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 1, kernel_size=(1, 1), stride=(1, 1)),
        )
        self.features3=nn.Sequential(
            nn.Conv2d(100,100,kernel_size=(1,8),stride=(1,1)),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 1, kernel_size=(1, 1), stride=(1, 1)),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(100, 1, 1, stride=1),  # 8*20
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(1, 8), stride=(1, 1))
        )
        self.classfiar=nn.Sequential(
            nn.Linear(in_features=8*2+20*2, out_features=2)
        )
    def forward(self,x):
        x1=self.features1(x.permute((0,3,1,2))) #100*8*20
        x2=(self.features2(x.permute((0,3,1,2)))) #100*8*20
        x3=(self.features3(x.permute((0,3,2,1))))  #100*20*8
        x4=(self.features3(x.permute((0,3,2,1))))
        x=torch.cat((x1,x2,x3,x4),2)
        x = x.view(-1, 56)
        x=self.classfiar(x)
        return x

#两层FCN
class Linear_2(nn.Module):
    def __init__(self,isDrop=True,p=0.2):
        super(Linear_2, self).__init__()
        self.isDrop=[isDrop,p]
        self.name='Linear_2'
        if self.isDrop:
            self.features=nn.Sequential(
                nn.Linear(8*20*100,4096),
                nn.ReLU(inplace=True),
                nn.Dropout(self.isDrop[1]),
                nn.Linear(4096, 2),
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(in_features=8 * 20 * 100, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=2),
            )
    def forward(self,x):
        x=x.view(-1,8*20*100)
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
                nn.Linear(8*20*100,8192),
                nn.ReLU(inplace=True),
                nn.Dropout(self.isDrop[1]),
                nn.Linear(8192, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 2),
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(8 * 20 * 100, 8192),
                nn.ReLU(inplace=True),
                nn.Linear(8192, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 2),
            )
    def forward(self,x):
        x=x.view(-1,8*20*100)
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
        elif self.net=='ConvNet':
            Model = ConvNet()
        if self.pretrained:
            Model.load_state_dict(torch.load(self.Weight_path))
        return Model

if __name__=='__main__':
    model=Model(Weight_path='./Weights/best_Linear_0_55.pth',net='ConvNet')
    transform = transforms.Compose([transforms.ToTensor()])
    #d = dataloader(path='./data', transforms=transform)
    #feature,label=d[0]
    #测试
    y=model.Net()
    print(y.name)
    print(y.features1)
