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


#2D卷积
class ConvNet_2D(nn.Module):
    def __init__(self):
        super(ConvNet_2D, self).__init__()
        self.name="ConvNet_2D"
        self.feature=nn.Sequential(
            nn.Conv2d(100, 200, 3, stride=1),
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(200, 400, 3, stride=1),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
        )
        #self.avger=nn.AdaptiveAvgPool2d((1, 1))  #TODO:看看后续有没有需要
        self.classfiar = nn.Sequential(
            nn.Linear(in_features=400*4*16, out_features=2)
        )
    def forward(self,x):
        # print("0000000000")
        # print(x)
        x1=self.feature(x.permute((0,3,1,2))) #100*8*20
        x1 = x1.reshape(-1, 400*4*16)
        x=self.classfiar(x1)

        return x
class ConvNet_sigmoid(nn.Module):
    def __init__(self):
        super(ConvNet_sigmoid, self).__init__()
        self.name='ConvNet_sigmoid'
        self.features1 = nn.Sequential(
            nn.Conv2d(100, 100, 1, stride=1),  #8*20
            nn.BatchNorm2d(100),
            nn.Sigmoid(),
            nn.Conv2d(100, 100, kernel_size=(1, 20), stride=(1, 1)),
            nn.Sigmoid(),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(100, 100,kernel_size=(1,20), stride=(1,1)),  # 8*20
            nn.BatchNorm2d(100),
            nn.Sigmoid(),
        )
        self.features3=nn.Sequential(
            nn.Conv2d(100,100,kernel_size=(1,8),stride=(1,1)),
            nn.BatchNorm2d(100),
            nn.Sigmoid(),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(100, 100, 1, stride=1),  # 8*20
            nn.BatchNorm2d(100),
            nn.Sigmoid(),
            nn.Conv2d(100, 100, kernel_size=(1, 8), stride=(1, 1)),
            nn.Sigmoid(),
        )
        self.classfiar=nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=56*100, out_features=2)
        )
    def forward(self,x):
        # print("0000000000")
        # print(x)
        x1=self.features1(x.permute((0,3,1,2))) #100*8*20
        # print("111111111")
        # print(x1)
        x2=(self.features2(x.permute((0,3,1,2)))) #100*8*20
        x3=(self.features3(x.permute((0,3,2,1))))  #100*20*8
        x4=(self.features3(x.permute((0,3,2,1))))
        x=torch.cat((x1,x2,x3,x4),2)
        x = x.view(-1, 56*100)
        # print(">>>>>>>>>>")
        # print(x)
        x=self.classfiar(x)
        # print("-----")
        # print(x)
        return x
class ConvNet(nn.Module):
    def __init__(self,num_class):
        super(ConvNet, self).__init__()
        self.name='ConvNet'
        self.features1 = nn.Sequential(
            nn.Conv2d(100, 100, 1, stride=1),  #8*20
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 100, kernel_size=(1, 20), stride=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(100, 100,kernel_size=(1,20), stride=(1,1)),  # 8*20
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
        )
        self.features3=nn.Sequential(
            nn.Conv2d(100,100,kernel_size=(1,8),stride=(1,1)),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(100, 100, 1, stride=1),  # 8*20
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 100, kernel_size=(1, 8), stride=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.classfiar=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=56*100, out_features=2)
        )
        self.classfiar_as = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=56 * 100+2, out_features=num_class),
        )
    def forward(self,x):
        # print("0000000000")
        # print(x)
        #TODO:这里引入了年龄和性别信息
        x_f=x[0]  #这里是纤维素的信息 （8，20，100）
        x_as=x[1]   #这里是性别和年龄的信息[2,1]
        x1=self.features1(x_f.permute((0,3,1,2))) #100*8*20
        # print("111111111")
        # print(x1)
        x2=(self.features2(x_f.permute((0,3,1,2)))) #100*8*20
        x3=(self.features3(x_f.permute((0,3,2,1))))  #100*20*8
        x4=(self.features3(x_f.permute((0,3,2,1))))
        x=torch.cat((x1,x2,x3,x4),2)
        x = x.view(-1, 56*100)
        # print(">>>>>>>>>>")
        # print(x)
        #TODO:引入年龄和性别特征
        age_sex=True
        if age_sex:
            x_as=x_as.view(-1,2)
            x=torch.cat((x,x_as),1)
            x=self.classfiar_as(x)
        else:
            x=self.classfiar(x)
        # print("-----")
        # print(x)
        return x
#去除第六和第七列
class ConvNet_18(nn.Module):
    def __init__(self,num_class):
        super(ConvNet_18, self).__init__()
        self.name='ConvNet_18'
        self.features1 = nn.Sequential(
            nn.Conv2d(100, 100, 1, stride=1),  #8*18
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 100, kernel_size=(1, 18), stride=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(100, 100,kernel_size=(1,18), stride=(1,1)),  # 8*20
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
        )
        self.features3=nn.Sequential(
            nn.Conv2d(100,100,kernel_size=(1,8),stride=(1,1)),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv2d(100, 100, 1, stride=1),  # 8*20
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 100, kernel_size=(1, 8), stride=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.classfiar=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=52*100, out_features=2)
        )
        self.classfiar_as = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=52 * 100+2, out_features=num_class)
        )
    def forward(self,x):
        # print("0000000000")
        # print(x)
        #TODO:这里引入了年龄和性别信息
        x_f=x[0] #这里是纤维素的信息 （8，20，100）
        x_as=x[1]   #这里是性别和年龄的信息[2,1]
        x1=self.features1(x_f.permute((0,3,1,2))) #100*8*20
        # print("111111111")
        # print(x1)
        x2=(self.features2(x_f.permute((0,3,1,2)))) #100*8*20
        x3=(self.features3(x_f.permute((0,3,2,1))))  #100*20*8
        x4=(self.features3(x_f.permute((0,3,2,1))))
        x=torch.cat((x1,x2,x3,x4),2)
        x = x.view(-1, 52*100)
        # print(">>>>>>>>>>")
        # print(x)
        #TODO:引入年龄和性别特征
        age_sex=True
        if age_sex:
            x_as=x_as.view(-1,2)
            x=torch.cat((x,x_as),1)
            x=self.classfiar_as(x)
        else:
            x=self.classfiar(x)
        # print("-----")
        # print(x)
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
class Linear_Sig_3(nn.Module):
    def __init__(self,isDrop=True,p=0.2):
        super(Linear_Sig_3, self).__init__()
        self.isDrop=[isDrop,p]
        self.name='Linear_Sig_3'
        if self.isDrop:
            self.features=nn.Sequential(
                nn.Linear(8*20*100,8192),
                nn.Sigmoid(),
                nn.Dropout(self.isDrop[1]),
                nn.Linear(8192, 4096),
                nn.Sigmoid(),
                nn.Dropout(0.5),
                nn.Linear(4096, 2),
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(8 * 20 * 100, 8192),
                nn.Sigmoid(),
                nn.Linear(8192, 4096),
                nn.Sigmoid(),
                nn.Linear(4096, 2),
            )
    def forward(self,x):
        x=x.view(-1,8*20*100)
        x=self.features(x)

        return x

#模型调用窗口
class Model:
    def __init__(self,num_class,Weight_path=' ',net='Linear_2',pretrained=False,isDrop=(False,0.2)):
        self.net=net
        self.pretrained=pretrained
        self.Weight_path=Weight_path
        self.isDrop=isDrop
        self.num_class=num_class

    def Net(self):
        if self.net=='Linear_2':
            Model=Linear_2(isDrop=self.isDrop)
        elif self.net=='Linear_3':
            Model = Linear_3(isDrop=self.isDrop)
        elif self.net=='ConvNet':
            Model = ConvNet(num_class=self.num_class)
        elif self.net=='ConvNet_18':
            Model = ConvNet_18(num_class=self.num_class)
        elif self.net=="Linear_Sig_3":
            Model = Linear_Sig_3()
        elif self.net=="ConvNet_2D":
            Model = ConvNet_2D()
        elif self.net == 'ConvNet_sigmoid':
            Model = ConvNet_sigmoid()
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
