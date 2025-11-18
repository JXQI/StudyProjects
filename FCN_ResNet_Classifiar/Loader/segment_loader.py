'''
Function: 加载数据集
Args: None
'''
import torch
from os.path import join
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import random
from matplotlib import pyplot as plt
import numpy as np

class_map=[0,255]   #对应的索引为其类别，0代表背景，1代表病变区域
'''
Function: label转化为类别信息
Args: Image.open()的label
'''
def image2label(label):
    label=np.array(label)
    label_class=np.zeros(label.shape)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            label_class[i][j]=class_map.index(label[i][j])
    return label_class
'''
Function: predict转化为图像信息
Args: predict的tensor
'''
def label2image(predict):
    predict=predict.cpu()
    predict=np.array(predict)
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            predict[i][j]=class_map[predict[i][j]]
    return predict

'''
Function: 随机裁剪图像
Args:
    label,data 均为Image.open()的值
    crop_size 剪裁的尺寸
'''
def random_crop(data,label,crop_size):
    height,width=crop_size
    height1=random.randint(0,data.size[0]-height)
    width1=random.randint(0,data.size[1]-width)
    height2=height1+height
    width2=width1+width
    data = data.crop((height1, width1, height2, width2))
    label = label.crop((height1, width1, height2, width2))
    return data,label
'''
Function: 可视化读取的图像
Args ：
'''
def show(image_name,label_name,Is_open=False):
    if not Is_open:
        image=Image.open(image_name)
        label=Image.open(label_name)
    else:
        image=image_name
        label=label_name
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(label,cmap='gray')
    plt.show()
    print(image.size)
    print(label.size)

'''
Funciton: 定义transforms
Args:   
    image:PIL对象
    label:PIL对象
    crop_size:()剪裁的大小
    transform:对image做的增强
Return:
    返回增强后的image和（h,w）的label，均为tensor类型
'''
def self_transforms(image,label,crop_size,transforms):
    image,label=random_crop(image,label,crop_size)
    image=transforms(image)
    label=image2label(label)
    label=np.array(label,dtype='int64')
    label=torch.from_numpy(label)

    return image,label

'''
Function: 数据集加载
Args:
    path: 数据集所在路径
    dataset: 需要加载的数据集，train or val
    transform: 图像增强  
'''
class dataloader(Dataset):
    def __init__(self,path,dataset,transform=None,crop_size=(500,500)):
        self.path=path
        self.dataset=dataset
        self.transform=transform
        self.crop_size=crop_size

        #存储图像和标签
        self.classfiar=["benign","malignant"]
        self.image_name=[]  #用于后续查看测试结果
        self.image=[]
        self.label=[]

        #直接将图像加载到内存里边
        file=pd.read_csv(join(dataset+'.csv'))
        tbar=tqdm(file["image_name"])
        print("\n加载数据:\n")
        for i,_ in enumerate(tbar):
            if self.transform:
                try:
                    image_name = join(path, "Images", file["image_name"][i] + '.jpeg')
                    label_name=join(path,"data",file["image_name"][i]+'_expert.png')
                    self.image_name.append(file["image_name"][i])
                    self.image.append(Image.open(image_name))
                    self.label.append(Image.open(label_name))
                except:
                    pass
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        image,label=self_transforms(self.image[item], self.label[item], crop_size=self.crop_size, transforms=self.transform)
        return image,label,self.image_name[item]

if __name__=='__main__':
    dataset=dataloader("../Data",dataset='segment_train',transform=transforms.Compose([transforms.ToTensor()]),crop_size=(500,500))
    print(len(dataset))
    val_dataset = dataloader("../Data", dataset='segment_val', transform=transforms.Compose([transforms.ToTensor()]))
    image,label,name=dataset[0]
    print(image.shape)
    print(label.shape)
    print(name)
    # ##测试各个函数的功能
    # image=join("../Data/Images",name+".jpeg")
    # label=join("../Data/data",name+"_expert.png")
    # show(image,label)
    # image,label=random_crop(data=Image.open(image),label=Image.open(label),crop_size=(500,500))
    # show(image,label,Is_open=True)
    # label_class=image2label(label)