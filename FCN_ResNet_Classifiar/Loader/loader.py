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


'''
Function: 数据集加载
Args:
    path: 数据集所在路径
    dataset: 需要加载的数据集，train or val
    transform: 图像增强  
'''
class dataloader(Dataset):
    def __init__(self,path,dataset,transform=None):
        self.path=path
        self.dataset=dataset
        self.transform=transform

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
            try:
                image_name=join(path,"Images",file["image_name"][i]+'.jpeg')
                self.image_name.append(file["image_name"][i])
                self.image.append(Image.open(image_name))
                self.label.append(self.classfiar.index(file["target"][i]))
            except:
                pass
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        return self.transform(self.image[item]),torch.tensor(self.label[item]),self.image_name[item]

if __name__=='__main__':
    dataset=dataloader("../Data",dataset='train',transform=transforms.Compose([transforms.ToTensor()]))
    print(len(dataset))
    val_dataset = dataloader("../Data", dataset='val', transform=transforms.Compose([transforms.ToTensor()]))
    print(len(val_dataset))

