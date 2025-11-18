import numpy as np
import os
import SimpleITK as sitk
from os.path import join
from PIL import Image
import pandas as pd
from skimage.measure import label,regionprops
from matplotlib import pyplot as plt
import random
import math
import shutil
from setting import DATA_PATH,LABEL_PATH,CSV1,CSV2

windows = 1400
leval = 600
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
function : 将标签的序号转换为类别信息
"""
def _class(name,label_id,df):
    key = df[(df.public_id == name) & (df.label_id == label_id)].index.tolist()
    label_code = df["label_code"][key[0]]  #TODO:这里添加类别有问题
    # 当类别=-1时，为了方便在mask中表示，这里表示为5，表示类型不清楚
    if label_code == -1: label_code = 5

    return label_code
"""
function : 统计每个切片的数据分布情况
"""
# 统计每个类的数目，病变区域的最值和众数
static_metric = {"class": {"1": 0, "2": 0, "3": 0, '4': 0, '5': 0}, "area":[],"bbox_area":[]}
class_name=["Displaced","Nondisplaced","Buckle","Segmental","Ignore"]
def static(regions):
    global static_metric
    print(regions)
    for region in regions:
        label_id,label_code,area,bbox_area=region
        static_metric["class"][str(label_code)]+=1
        static_metric["area"].append(area)
        static_metric["bbox_area"].append(bbox_area)
"""
function : 对统计结果进行分析
"""
def analysis(plot=True):
    global static_metric
    global class_name

    info={"class":dict(),"area_min":0,"area_max":0,"area_median":0,\
          "bbox_min":0,"bbox_max":0,"bbox_median":0}
    area = pd.Series(static_metric["area"])
    bbox_area=pd.Series(static_metric["bbox_area"])
    info["class"]=static_metric["class"]
    info["area_min"]=area.min()
    info["area_max"]=area.max()
    info["area_median"]=area.median()
    info["bbox_min"]=bbox_area.min()
    info["bbox_max"]=bbox_area.max()
    info["bbox_median"]=bbox_area.median()

    # 是否画直方图
    if plot:
        # 不同类型骨折的数目
        plt.figure(1)
        plt.title("The number of different rifrac")
        plt.bar(class_name, info["class"].values())
        for a, b in zip(class_name, info["class"].values()): # 显示数目
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        plt.savefig("number_data.png")
        plt.figure(2)
        # area和bbox的值
        plt.title("area and bbox_area")
        name_list=["min","max","median"]
        num_list1=[info["area_min"],info["area_max"],info["area_median"]]
        num_list2=[info["bbox_min"],info["bbox_max"],info["bbox_median"]]
        x = list(range(len(num_list1)))
        total_width, n = 0.8, 2
        width = total_width / n

        plt.bar(x, num_list1, width=width, label='area', fc='y')
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, num_list2, width=width, label='bbox', tick_label=name_list, fc='r')
        plt.legend()
        # 显示或者存储
        plt.savefig("area_bbox.png")
        plt.show()

    return info

"""
function : 上一个产生切片的方式过于粗暴：
        1.对每个切片中不为0的部分单独考虑，单独计算面积，只保留大于设定阈值的部分，并且mask的值代表的就是标签值
        2.对于多个不连续的区域直接根据根据队列做差的方式判断，从而分成多个区域
        3.根据矩形框面积的大小来进行筛选，但是实例的形状是不规则的，矩形框不一定能够代表真正的区域大小
        
        虽然实现了功能，但是有更简单的方法，利用skimage的库来进行操作
"""
def slices_3(dataname_list,dataset_type=True):
    global class_name
    data_path = DATA_PATH
    label_path = LABEL_PATH
    label_list=[file.split("-")[0]+"-label.nii.gz" for file in dataname_list]

    # 数据保存路径
    data_dir = './data'
    train_dir = './data/coco/train2017'
    val_dir = "./data/coco/val2017"
    test_dir = "./data/coco/test2017"
    mask_dir = './data/coco/mask'
    annotations_dir = './data/coco/annotations'

    if not os.path.isdir(data_dir):
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        # os.makedirs(test_dir)
        os.makedirs(mask_dir)
        os.makedirs(annotations_dir)

    #读取rifrac-train-info-1.csv文件，确定类别信息
    df1=pd.read_csv(CSV1)
    df2=pd.read_csv(CSV2)
    df=pd.concat([df1,df2],ignore_index=True)
    #训练数据的大小
    #number=len(dataname_list)
    number=1
    for i in range(number):
        i=8
        nii=sitk.ReadImage(join(data_path,dataname_list[i]))
        image=sitk.GetArrayFromImage(nii)
        nii_label=sitk.ReadImage(join(label_path,label_list[i]))
        image_label=sitk.GetArrayFromImage(nii_label)
        print("nii:{} label:{} nii_shape{},label_shape{}".format(dataname_list[i],label_list[i],image.shape,image_label.shape))
        name=dataname_list[i].split('-')[0]

        # 每个切片的值
        axial,axial_label=image,image_label   #z,height,width
        coronal,coronal_label=image.transpose(1,2,0),image_label.transpose(1,2,0)
        sagital,sagital_label=image.transpose(2,1,0),image_label.transpose(2,1,0)

        for j in range(axial_label.shape[0]):
            slice=axial_label[j]
            # 如果存在实例则进行判断是否保存为实例的标签
            if np.any(slice):
                # 对切片图像进行连通操作，对同一个连通域进行编号
                slice_label=label(slice>0)
                # 求不同连通域最大的像素值从而来确定当前连通域的类别，同时根据此连通域的面积决定是否保留当前连通域
                slice_regions=regionprops(slice_label,slice)
                slice_all_regions=[(region.label,_class(name,region.max_intensity,df),region.area,region.bbox_area) for region in slice_regions]
                # 统计样本数据分布情况
                # static(slice_all_regions)
                # 保存mask数据
                index=0
                for k in range(len(slice_all_regions)):
                    label_id, label_code, area, bbox_area = slice_all_regions[k]
                    slice_name=name+'_'+str(j+1)+".png"
                    slice_axial = axial[j]
                    if area>250:
                        mask_name=name+'_'+str(j+1)+"_"+class_name[label_code-1]+'_'+str(index)+'.png'
                        index+=1
                        mask_image=slice_label==label_id
                        # 保存image和mask
                        # 转换成RGB和gray图像
                        slice_axial = Image.fromarray(
                            window_transform3(slice_axial, windowWidth=windows, windowCenter=leval)).convert('RGB')
                        slice_axial_label = Image.fromarray(mask_image).convert('L')

                        # 判断保存的位置，如果为真则保存在train2017下边
                        if dataset_type:
                            slice_axial.save(join(train_dir,slice_name))
                        else:
                            slice_axial.save(join(val_dir,slice_name))
                        slice_axial_label.save(join(mask_dir,mask_name))


"""
function : 对数据集进行划分，分为训练集和验证集
"""
def divide(path,ratio):
    image_dir=join(path,'image')
    label_dir=join(path,'mask')
    # 数据保存路径
    data_dir = './data'
    train_dir = './data/coco/train2017'
    val_dir = "./data/coco/val2017"
    test_dir = "./data/coco/test2017"
    mask_dir = './data/coco/mask'
    annotations_dir = './data/coco/annotations'

    # 先清空各个数据集，重新写入
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    if not os.path.isdir(data_dir):
        os.makedirs(train_dir)
        os.makedirs(val_dir)
        # os.makedirs(test_dir)
        os.makedirs(mask_dir)
        os.makedirs(annotations_dir)

    # 划分数据集
    image_list=os.listdir(image_dir)
    print(image_list)
    N=len(image_list)
    # 打乱两次
    random.shuffle(image_list)
    random.shuffle(image_list)
    train=image_list[:math.ceil(N*ratio)]
    val=image_list[math.ceil(N*ratio):]
    print(N,len(train),len(val))

    # 将数据集移到对于的文件夹下
    for image in train:
        file=os.path.join(image_dir,image)
        shutil.move(file,train_dir)
    for image in val:
        file = os.path.join(image_dir, image)
        shutil.move(file, val_dir)
    # 将mask图像移动到mask_dir下边
    for mask in os.listdir(label_dir):
        file=join(label_dir,mask)
        shutil.move(file,mask_dir)

    # 删除原来的保存目录
    shutil.rmtree(path)

    print("划分数据集完成")

"""
fucntion : 对nii.gz文件直接进行划分
args : 输入是数据集的路径
"""
def divide_nii(path,ratio):
    filelist=[file for file in os.listdir(path) if file.endswith('.gz')]
    random.shuffle(filelist)
    N=len(filelist)
    train=filelist[:math.ceil(N*ratio)]
    val=filelist[math.ceil(N*ratio):]
    print(len(train),len(val),N)
    return train,val

if __name__=='__main__':
    train,val=divide_nii(DATA_PATH,5/6)

    # 先清空各个数据集，重新写入
    if os.path.isdir('./data'):
        shutil.rmtree('./data')
    # 生成切片数据，保存在train2017下边
    slices_3(train,dataset_type=True)
    # 生成val数据集，保存在val2017下边
    slices_3(val,dataset_type=False)

    # 显示统计数据分布
    # info=analysis()
    # print(info["class"].keys(),info["class"].values())
    # print(info)

    # 划分训练集和验证集，会创建新的目录data,删除原来的文件夹
    # path="./axial_slice"
    # divide(path,5/6)

