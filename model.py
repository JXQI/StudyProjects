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

class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.fc1=nn.Linear(8*20*100,4096)
        self.fc2=nn.Linear(4096,2)
    def forward(self,x):
        x=x.view(-1,8*20*100)
        #print("--------")
        #print(x)
        #x=self.fc1(x)
        x=F.relu(self.fc1(x))
        #x=F.sigmoid(self.fc1(x))
        #print("**********")
        #print(x)
        x=self.fc2(x)
        #print("||||||||||")
        #x=F.softmax(x,dim=1)
        #print(x)

        return x
if __name__=='__main__':
    model=Linear()
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
    y=model(feature)
    print(y)
