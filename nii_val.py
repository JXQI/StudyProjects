"""
function: 直接用来输入nii.gz格式的图片用于测试
"""
import numpy as np
import os
import glob
from scipy import ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision.transforms as transforms
from os.path import join
from PIL import Image
from settings import HU_WINDOW,HU_LEVEL
import transforms as T
import utils
from model import get_model_instance_segmentation
from  settings import NUM_CLASS,GPU
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
function: 针对.nii.gz格式的加载文件
"""
class NiiDataset(object):
    def __init__(self, data, label,transforms):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = data
        self.labels=label
    def __getitem__(self, idx):
        #先对数据加窗
        img=Image.fromarray(window_transform(self.imgs[idx],windowWidth=HU_WINDOW,windowCenter=HU_LEVEL)).convert('RGB')
        # load images ad masks
        img,label=self.transforms(img,self.labels[idx])
        return img,label

    def __len__(self):
        return len(self.imgs)

'''
function: 加窗HU,Leval----Windows
Args: ct-array: .nii.gz格式的CT图像
      windowWidth,windowCenter:  窗宽，窗中心
      nomal: 为True时为16为整型tif，False时为0-255为RGB
Return: 
'''
def window_transform(ct_array, windowWidth, windowCenter, normal=False):
   maxWindow=(2*windowCenter+windowWidth)/2
   minWindow=(2*windowCenter-windowWidth)/2
   newimg=ct_array.astype("int16")
   newimg[newimg < minWindow] = minWindow
   newimg[newimg > maxWindow] = maxWindow
   if not normal:
       newimg = (newimg * 255).astype('uint8')
   return newimg

'''
function: 模型加载
return: 各个截面对应的模型
'''
def DetectionModel(data,label, modelpt):
    dataset = NiiDataset(data,label,get_transform())
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    model = get_model_instance_segmentation(NUM_CLASS)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not GPU:
        model.load_state_dict(torch.load(modelpt,map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(modelpt))
    # move model to the right device
    model.to(device)

    return data_loader, model

"""
function: 输出预测图，并且生成nii.gz文件
return: 返回在原图上标注预测结果的图像
"""
def prediction(dataloader, model):
    nii = []
    nii_mask=[]
    nii_label=[]
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(dataloader):
            print(i,len(dataloader))
            predict = model(data[0])
            pres = np.zeros(predict[0]['masks'].shape[1:])
            for i in range(predict[0]['masks'].shape[0]):
                # pres += predict[0]['masks'][i, 0].mul(
                #     (i + 1) * 255 // predict[0]['masks'].shape[0]).byte().cpu().numpy()
                temp=np.array(np.ceil(predict[0]['masks'][i, 0]))
                temp[temp==1]=i+1
                pres+=temp
            mask=pres[0]
            # img=Image.fromarray(np.array(data[0][0].permute(1,2,0)),mode='RGB').convert('L')
            # img=np.array(img)
            # nii.append(img)
            label=np.array(data[1][0])

            nii_label.append(label)
            nii_mask.append(mask)
    return np.array(nii_label,dtype="int16"),np.array(nii_mask,dtype='int16')

"""
function: 用于单个截面处理单个nii.gz的文件，并且最后预测并且保存成nii.gz格式的文件
Args: 输入的单个nii.gz文件
"""
def signal_nii(niiname,nii_savepath,axial,label,modelpt):
    print("预测数据的大小:{}".format(axial.shape))
    #生成预测的结果
    dataloader,model=DetectionModel(data=axial,label=label,modelpt=modelpt)
    data,premask=prediction(dataloader,model)
    print("预测结果大小:{}".format(premask.shape))
    nii_maskfile=sitk.GetImageFromArray(premask)
    nii_file=sitk.GetImageFromArray(data)

    #必须设置方向，不然和原图像在ITK中打开对不齐
    nii_file.SetDirection(driection)
    nii_file.SetSpacing(space)
    nii_file.SetOrigin(origin)
    nii_maskfile.SetDirection(driection)
    nii_maskfile.SetOrigin(origin)
    nii_maskfile.SetSpacing(space)
    sitk.WriteImage(nii_file, join(nii_savepath, "signal_" + niiname))
    sitk.WriteImage(nii_maskfile,join(nii_savepath,"signal_mask_"+niiname))

"""
function: 利用三个截面图用于处理单个nii.gz的文件，并且最后预测并且保存成nii.gz格式的文件
Args: 输入的单个nii.gz文件
"""
def Decetion_nii(niipath,niiname,nii_savepath,modelpt):
    niidata=join(niipath,niiname)
    nii = sitk.ReadImage(niidata)
    axial=sitk.GetArrayFromImage(nii)
    coronal = axial.transpose(2, 1, 0)
    sagittal = axial.transpose(1, 2, 0)

    imgs=[]
    print(axial.shape)
    for i in range(axial.shape[0]):
        im=Image.fromarray(window_transform(axial[i],windowWidth=HU_WINDOW,windowCenter=HU_LEVEL,normal=True)).convert('RGB')
        #生成预测的结果
        # pre=prediction(DetectionModel(data=im,modelpt=modelpt))
        # print(pre.shape)
        #这块还需要将图像转存I格式，RGB为三通道不能直接保存
        im_np=np.array(im)
        imgs.append(im_np)
    imgs=np.array(imgs)
    nii_file=sitk.GetImageFromArray(imgs)

    nii_file.SetDirection(driection)
    nii_file.SetSpacing(space)
    nii_file.SetOrigin(origin)
    sitk.WriteImage(nii_file,join(nii_savepath,niiname))

def get_transform(train=False):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__=='__main__':
    nii_path="/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/ribfrac-val-images"
    label_path="/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/ribfrac-val-labels"
    nii_savepath="./nii_test"
    if not os.path.isdir(nii_savepath):
        os.mkdir(nii_savepath)
    nii_datalist=[i for i in os.listdir(nii_path) if i.endswith('.gz')]
    nii_datalist.sort()
    label_datalist=[i for i in os.listdir(label_path) if i.endswith('.gz')]
    label_datalist.sort()
    print("验证集大小：{}".format(len(nii_datalist)))

    niidata=join(nii_path,nii_datalist[0])
    labeldata=join(label_path,label_datalist[0])
    nii = sitk.ReadImage(niidata)
    space=nii.GetSpacing()
    origin=nii.GetOrigin()
    driection=nii.GetDirection()

    # masknii=sitk.ReadImage("./nii_test/signal_mask_RibFrac421-image.nii.gz")
    # masknii.SetSpacing(nii.GetSpacing())
    # masknii.SetOrigin(nii.GetOrigin())
    # print(masknii.GetSpacing())


    label= sitk.ReadImage(labeldata)
    axial=sitk.GetArrayFromImage(nii)
    axial_label=sitk.GetArrayFromImage(label)

    # axial=axial[118:120]
    # axial_label=axial_label[118:120]

    signal_nii(niiname=nii_datalist[0],nii_savepath=nii_savepath,axial=axial,label=axial_label,modelpt='./Weights/axial.pt')
    # signal_nii(nii_path,nii_datalist[0],nii_savepath)