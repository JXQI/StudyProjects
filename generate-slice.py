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
import pandas as pd

windows = 600
leval = 550
#加窗  不做归一化
def window_transform3(ct_array, windowWidth, windowCenter, normal=False):
   """
   return: trucated image according to window center and window width
   and normalized to [0,1]
   """
   maxWindow=windowCenter+windowWidth//2
   minWindow=windowCenter-windowWidth//2
   newimg=ct_array.astype("int16")
   newimg[newimg < minWindow] = minWindow
   newimg[newimg > maxWindow] = maxWindow
   if not normal:
       newimg = (newimg/windowWidth*255).astype('uint8')
   return newimg

"""
function: 将图像转化成相应的格式
"""
def nii2png(slice_axial,slice_coronal,slice_sagital,label=False):
    if not label:
        slice_axial = Image.fromarray(window_transform3(slice_axial, windowWidth=windows, windowCenter=leval)).convert('RGB')
        slice_coronal = Image.fromarray(window_transform3(slice_coronal, windowWidth=windows, windowCenter=leval)).convert('RGB')
        slice_sagital = Image.fromarray(window_transform3(slice_sagital, windowWidth=windows, windowCenter=leval)).convert('RGB')
    else:
        slice_axial = Image.fromarray(slice_axial).convert('L')
        slice_coronal= Image.fromarray(slice_coronal).convert('L')
        slice_sagital = Image.fromarray(slice_sagital).convert('L')
    return slice_axial,slice_coronal,slice_sagital

"""
function: 填充面积小于阈值的mask，直接填充为0
"""
def mask_fill(mask_new,loc):
    len_x = len(loc[0])
    for i in range(len_x):
        indice_x = loc[0][i]
        indice_y = loc[1][i]
        mask_new[indice_x][indice_y] = 0
    return mask_new

"""
function:处理多个切片中不连通的多个区域，如果只有一个直接返回，如果有多个不连通区域，只返回面积最大的部分
args: 输入位置x,y以及mask的类别,还有mask
"""
def remove(loc,c,mask_new):
    #计算所有区域合并起来的面积
    area_all=(np.max(loc[0])-np.min(loc[0]))*(np.max(loc[1])-np.min(loc[1]))
    #创建新的mask图像，保证每个图像中只有一个同类别的图像，为了和后边的mask-r-cnn的loader对应
    mask_new=np.zeros(mask_new.shape)
    #对区域进行划分
    x = loc[0]
    x_temp = list(loc[0][1:])
    x_temp.append(0)
    res = np.array(x_temp - x)  #错位做差，如果不连续相减肯定大于1
    if np.any(res>1):
        #记录下最大区域，并且记录下最大区域的坐标
        max_area,area_indice=0,((0,0),(0,0))
        #将整个坐标分成不连续的段，然后计算area
        indice=[-1]  #写成-1方便后边统一
        area_x=np.where(res>1)
        indice.extend(area_x[0])
        indice.extend([len(x_temp)-1])
        # print(len(loc[0]),indice)
        # print("------原来区域")
        # print(loc)
        for i in range(1,len(indice)):
            left=indice[i-1]+1
            right=indice[i]
            if not right==left:
                area=(loc[0][right]-loc[0][left])*(np.max(loc[1][left:right])-np.min(loc[1][left:right]))
            else:
                area=0
            if area>max_area:
                max_area=area
                area_indice=(left,right)
        # print("---{}_{}_{}----".format(area_all,area,max_area))
        #给新mask赋值
        len_x=area_indice[1]-area_indice[0]+1
        for i in range(len_x):
            begin=area_indice[0]
            indice_x=loc[0][begin+i]
            indice_y=loc[1][begin+i]
            mask_new[indice_x][indice_y]=c
        # print("------原来区域")
        # print(loc)
        # print("------去除后的区域")
        # print(np.where(mask_new==c))
    else:
        len_x=len(loc[0])
        for i in range(len_x):
            indice_x = loc[0][i]
            indice_y = loc[1][i]
            mask_new[indice_x][indice_y] = c

    return mask_new


"""
function: 叠加三个维度的正交图像，生成三个截面有标注的图像
          尝试：
            1.不保存为图像，直接保存为矩阵，因为图像会有信息值损失
            2.对mask的大小进行筛选，阈值需要实际看一下
"""
def slices_3():
    data_path = '/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/ribfrac-train-images/Part1'
    label_path = '/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/Part1'
    # data_path="/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/ribfrac-val-images"
    # label_path="/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/ribfrac-val-labels"

    dataname_list = os.listdir(data_path)
    dataname_list = [i for i in dataname_list if i.endswith('.gz')]
    #对列表进行排序，方便后边label对应，同时方便后边查看分割效果
    sort_index=[int(dataname_list[i].split('-')[0][7:]) for i in range(len(dataname_list))]
    dataname_list=[x for _,x in sorted(zip(sort_index,dataname_list))]

    label_list=[i for i in  os.listdir(label_path) if i.endswith('.gz')]
    sort_index=[int(label_list[i].split('-')[0][7:]) for i in range(len(label_list))]
    label_list=[x for _,x in sorted(zip(sort_index,label_list))]

    #读取rifrac-train-info-1.csv文件，确定类别信息
    df=pd.read_csv("/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/ribfrac-train-info-1.csv")
    #所有的类别信息，包含bg0总共6类，5代表-1
    class_name=[0,1,2,3,4,5]
    # key=df[(df.public_id=='RibFrac1')&(df.label_id==0)].index.tolist()
    # print(key)
    # print(df["label_code"][key[0]])
    #训练数据的大小
    #number=len(dataname_list)
    number=200
    for i in range(number):
        nii=sitk.ReadImage(join(data_path,dataname_list[i]))
        image=sitk.GetArrayFromImage(nii)
        nii_label=sitk.ReadImage(join(label_path,label_list[i]))
        label=sitk.GetArrayFromImage(nii_label)
        print("nii:{} label:{} nii_shape{},label_shape{}".format(dataname_list[i],label_list[i],image.shape,label.shape))

        # 获取mask类别信息
        name=dataname_list[i].split('-')[0]

        # 每个切片的值
        axial,axial_label=image,label   #z,height,width
        coronal,coronal_label=image.transpose(1,2,0),label.transpose(1,2,0)
        sagital,sagital_label=image.transpose(2,1,0),label.transpose(2,1,0)

        for j in range(axial_label.shape[0]):
            slice=axial_label[j]
            if np.any(slice):
                class_num=np.unique(slice)
                #后边要通过区域的面积，每次遍历都改变mask的值
                # mask_new=slice
                mask_new=np.zeros(slice.shape)
                flag=False  #判断该mask是否有大于阈值
                # print(class_num)
                for k in class_num:
                    if k!=0:
                        # 获取类别信息
                        label_id=k
                        key = df[(df.public_id == name) & (df.label_id == label_id)].index.tolist()
                        label_code=df["label_code"][key[0]] #类别信息
                        # 当类别=-1时，为了方便在mask中表示，这里表示为5，表示类型不清楚
                        if label_code==-1:label_code=5
                        # 根据mask的区域大小进行筛选
                        info={}
                        local=np.where(slice==k)
                        info["width"],info["height"]=np.max(local[0])-np.min(local[0]),np.max(local[1])-np.min(local[1])
                        info["center"]=(np.min(local[0])+info["width"]//2,np.min(local[1])+info["height"]//2)
                        info["area"]=info["height"]*info["width"]
                        info["indices"]=(j,info["center"][0],info["center"][1])
                        # print(info['area'],j)
                        if info['area']>350:
                            flag=True
                            # mask_new=remove(local, int(label_code), mask_new)
                            mask_new += remove(local, int(label_code), slice)
                        else:
                            # mask_new+=mask_fill(mask_new, local)
                            pass
                #判断生成的mask是否正确，否则抛出异常
                mask_class=np.unique(mask_new)
                assert set(mask_class).issubset(set(class_name))
                if flag:
                    print("改变mask后{}".format(mask_class))
                    # 保存image和mask
                    slice_axial = axial[j]
                    slice_axial_label = mask_new
                    # 转换成RGB和gray图像
                    slice_axial = Image.fromarray(window_transform3(slice_axial, windowWidth=windows, windowCenter=leval)).convert('RGB')
                    slice_axial_label = Image.fromarray(slice_axial_label).convert('L')
                    # 保存图像
                    path = "./axial_slice"
                    if not os.path.isdir(path):
                        os.makedirs(join(path, 'image'))
                        os.makedirs(join(path, 'mask'))
                    image_name = dataname_list[i].split('-')[0] + '_' + str(j) + '.png'
                    slice_axial.save(join(path,"image", image_name))
                    slice_axial_label.save(join(path,'mask', image_name))

if __name__=='__main__':
    slices_3()
    # path="./axial_slice/mask"
    # for i in os.listdir(path):
    #     img=Image.open(join(path,i))
    #     img=np.array(img)
    #     print(i)
    #     print(np.unique(img))