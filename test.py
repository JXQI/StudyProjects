import torch
from engine import train_one_epoch, evaluate
import utils
import torch
import transforms as T
from loader import PennFudanDataset
from model import get_model_instance_segmentation
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from random import random
from settings import IMAGE,MODEL_NAME,AXIAL_MODEL,CORNAL_MODEL,ATTENTION,LOG,DEBUG_LOG,\
    SAGIT_MODEL,AXIAL_TEST,CORNAL_TEST,SAGIT_TEST,NUM_CLASS,HU_LEVEL,HU_WINDOW,NII_GZ,NII_GZ_SAVE
import SimpleITK as sitk
from os.path import join
from PIL import ImageDraw
import global_attention as G
from global_attention import attention
#判断是否检测到
EXIST=False
OUTPUT,mask_OUTPUT=[],[] #保存nii数据和mask数据


#初始化全局变量
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
axial_dataset=None
cornal_dataset=None
sagit_dataset=None
#记录切片的位置
axial_slices=[]
cornal_slices=[]
sagit_slices=[]
axial_model=None
cornal_model=None
sagit_model=None

"""
function: 针对.nii.gz格式的加载文件
"""
class NiiDataset(object):
    def __init__(self, data,transforms):
        self.transforms=transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = data
        self.imgs_slices = [str(idx) for idx in range(len(self.imgs))]
    def __getitem__(self, idx):
        #先对数据加窗
        img=Image.fromarray(window_transform(self.imgs[idx],windowWidth=HU_WINDOW,windowCenter=HU_LEVEL)).convert('RGB')
        # load images ad masks
        if self.transforms:
            # 这里的label没有意义，只为了程序通过
            img,label=self.transforms(img,img)
        return img,label,idx

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

"""
function:画框并且保存数据
"""
def draw_save(boxes_pre,image_gray,N):
    global OUTPUT
    global mask_OUTPUT
    global EXIST

    mask_gray = Image.new('L', size=image_gray.size, color=0)
    if boxes_pre:
        EXIST = True
        for i in boxes_pre:
            x0, y0, x1, y1 = i
            # 在原图上画矩形框并且保存
            if LOG:print("====>在原图上标记检测框:{}".format(i))
            draw = ImageDraw.Draw(image_gray)
            draw.rectangle(i,fill=None,outline="black")
            #在mask上画矩形框并且保存
            if LOG:print("====>在mask上标记检测框:{}".format(i))
            #为mask着色
            mask_gray=np.array(mask_gray)
            mask_gray[int(x0):int(x1), int(y0):int(y1)] = boxes_pre.index(i) + 1
    OUTPUT.append(np.array(image_gray))
    mask_OUTPUT.append(np.array(mask_gray))
    if LOG:print("\n====>已经检测的切片数目：{}/{}<====\n".format(len(OUTPUT),N))
    # output=np.array(OUTPUT)
    # mask_output=np.array(mask_OUTPUT)
    # if LOG:print("输出图像的形状：{},mask的形状：{}".format(output.shape,mask_output.shape))
"""
function: 对多个nii切片做判断，并且加mask和boxes
args: 输入一个完整的切片图像arry，输出同样形状，并且加了mask和boxes的图像arry
"""
def prediction(nii_image):
    # 输出图像切片
    global OUTPUT
    global mask_OUTPUT
    N = len(axial_dataset)
    if LOG:print("切片个数为：{}".format(N))
    # 需要判断第几个数据
    for order in range(N):  #TODO:修改大小快速测试程序
        global EXIST
        EXIST = False
        img,label,index = axial_dataset[order]
        model = axial_model
        model.eval()
        with torch.no_grad():
            predict = model([img.to(device)])
            image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            #转化成灰度图最后保存
            image_gray = image.convert('L')
            #保存预测的mask
            mask_gray = Image.new('L',image_gray.size,color=0)
            boxes_pre = predict[0]['boxes']     #TODO:考虑后边加不加mask，等到效果验证好了再决定
            # 合并重叠的区域
            boxes_pre = inter_rec(boxes_pre)
            # 计算boxes中心
            boxes_center = [box_center(i) for i in boxes_pre]
            if LOG:print("axial截面检测到的boxes中心{}".format(boxes_center))
            boxes_pre = [list(i) for i in boxes_pre if judge_cor_sig(i, index)]
            if LOG:print("正交判断的结果：{}".format(boxes_pre))

            if ATTENTION:  #如果存在相邻切片直接的判断，则进行判断，否则直接保存检测到的单个切片
                attention(image_gray,boxes_pre,N)
                nii_output, nii_mask_output = np.array(G.output), np.array(G.mask_output)  # global_attention中的全局变量
            else:
                draw_save(boxes_pre,image_gray,N)
                nii_output, nii_mask_output = np.array(OUTPUT), np.array(mask_OUTPUT)  # 本文件中的全局变量
    #清空判断下一个nii数据
    G.output, G.mask_output = [], []
    OUTPUT,mask_OUTPUT=[],[]
    return nii_output,nii_mask_output
    # #注释部分功能正常，实现保存和画框功能
    #         if boxes_pre:
    #             EXIST = True
    #
    #             # TODO:这里是在plt中的显示函数，没有其他作用
    #             pres = np.zeros(predict[0]['masks'].shape[1:])
    #             # print("类别数目:{}".format(len(prediction[0]["masks"])))
    #             for i in range(predict[0]['masks'].shape[0]):
    #                 # print(np.unique(np.array(np.ceil(prediction[0]['masks'][i, 0]),dtype='int16')))
    #                 pres += predict[0]['masks'][i, 0].mul(
    #                     (i + 1) * 255 // predict[0]['masks'].shape[0]).byte().cpu().numpy()
    #             pre = Image.fromarray(pres[0])
    #             fig = plt.figure()
    #             ax1 = fig.add_subplot(2, 2, 1)
    #             plt.title('slice_image')
    #             plt.imshow(image)
    #             ax2 = fig.add_subplot(2, 2, 2)
    #             plt.title('mask')
    #             plt.imshow(pre)
    #             ax3 = fig.add_subplot(2, 2, 3)
    #             for i in boxes_pre:
    #                 x0, y0, x1, y1 = i
    #                 # color = (random(), random(), random())
    #                 # rect = plt.Rectangle((x0, y0), abs(x1 - x0), abs(y1 - y0), edgecolor=color, fill=False, linewidth=1)
    #                 # ax3.add_patch(rect)
    #                 # 在原图上画矩形框并且保存
    #                 print("====>在原图上标记检测框:{}".format(i))
    #                 draw = ImageDraw.Draw(image_gray)
    #                 draw.rectangle(i,fill=None,outline="black")
    #                 #在mask上画矩形框并且保存
    #                 print("====>在mask上标记检测框:{}".format(i))
    #                 #为mask着色
    #                 mask_gray=np.array(mask_gray)
    #                 mask_gray[int(x0):int(x1), int(y0):int(y1)] = boxes_pre.index(i) + 1
    #             plt.imshow(image)
    #         output.append(np.array(image_gray))
    #         mask_output.append(np.array(mask_gray))
    #         print("\n====>已经检测的切片数目：{}/{}<====\n".format(len(output),N))
    # output=np.array(output)
    # mask_output=np.array(mask_output)
    # print("输入图像的形状：{},输出图像的形状：{},mask的形状：{}".format(nii_image.shape, output.shape,mask_output.shape))
    # return output,mask_output

"""
function: 用于单个截面处理单个nii.gz的文件，并且最后预测并且保存成nii.gz格式的文件
Args: 输入的单个nii.gz文件
"""
def signal_nii(nii,nii_savepath):
    global axial_dataset, cornal_dataset, sagit_dataset
    global axial_slices, cornal_slices, sagit_slices

    #读取nii数据
    niidata = sitk.ReadImage(nii)
    space = niidata.GetSpacing()
    origin = niidata.GetOrigin()
    driection = niidata.GetDirection()

    nii_image=sitk.GetArrayFromImage(niidata)
    # 初始化dataloder
    set_transforms=get_transform(train=False)
    axial_dataset = NiiDataset(nii_image,set_transforms)
    cornal_dataset = NiiDataset(nii_image.transpose(1,2,0),set_transforms)
    sagit_dataset = NiiDataset(nii_image.transpose(2, 1, 0),set_transforms)

    axial_slices = axial_dataset.imgs_slices
    cornal_slices = cornal_dataset.imgs_slices
    sagit_slices = sagit_dataset.imgs_slices

    #返回原图上带有boxes和mask的切片
    pre_data,pre_mask=prediction(nii_image)
    print(pre_data.shape,pre_mask.shape)
    pre_nii_file=sitk.GetImageFromArray(pre_data)
    pre_nii_mask=sitk.GetImageFromArray(pre_mask)
    #必须设置方向，不然和原图像在ITK中打开对不齐
    pre_nii_file.SetDirection(driection)
    pre_nii_file.SetSpacing(space)
    pre_nii_file.SetOrigin(origin)

    pre_nii_mask.SetDirection(driection)
    pre_nii_mask.SetSpacing(space)
    pre_nii_mask.SetOrigin(origin)
    sitk.WriteImage(pre_nii_file, join(nii_savepath, "image" , nii.split("/")[-1]))
    sitk.WriteImage(pre_nii_mask, join(nii_savepath, "mask" , nii.split("/")[-1]))

"""
function: 做数据增强
"""
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

"""
function:计算所有boxes的中心位置
args: box(x0,y0,x1,y1)
"""
def box_center(box):
    x=int(abs(box[2]-box[0])//2+box[0])
    y=int(abs(box[3]-box[1])//2+box[1])
    return (x,y)
"""
function: 判断两个矩形是否相交，如果相交合并返回最小的矩形外接框
"""
def inter_rec(boxes_pre):
    #将boxes根据x1的大小进行排序
    indice = boxes_pre[:, 0]
    boxes_pre_new = []
    try:
        for _, box in sorted(zip(indice, boxes_pre),key=lambda x:(x[0],x[1][1])):
            boxes_pre_new.append(np.array(box.cpu()))
        boxes_pre_new = np.array(boxes_pre_new)
        #x0,y0,x1,y1
        i=0
        boxes_pre_new=list(boxes_pre_new)
        while(i<len(boxes_pre_new)-1):
            x0,y0,x1,y1=boxes_pre_new[i]

            j=i+1 #从当前位置的下一个元素开始判断
            Flag=False #是否有重合的boxes
            while(j<len(boxes_pre_new)):
                m0, n0, m1, n1 = boxes_pre_new[j]
                if not (m0>x1 or n0>y1 or n1<y0):
                    t0,z0=x0,min(y0,n0)
                    t1,z1=max(x1,m1),max(y1,n1)
                    box=[t0,z0,t1,z1]
                    boxes_pre_new[i]=np.array(box)  #因为box是最小包含矩形框，所以这里将i位置的替换，而将j位置的去掉，这样可以维持原来boxes的顺序
                    boxes_pre_new.pop(j)
                    Flag=True
                    break
                else:
                    j+=1
            if not Flag:
                i=i+1
    except:
        print("判断矩形相交部分出错！！！")
    return boxes_pre_new

"""
function: 初始化加载各种数据，各种模型
"""
def Init(label=True):
    global axial_dataset,cornal_dataset,sagit_dataset
    global axial_model,cornal_model,sagit_model
    global axial_slices,cornal_slices,sagit_slices
    # #初始化dataloder
    axial_dataset=PennFudanDataset(AXIAL_TEST, get_transform(train=False))
    cornal_dataset=PennFudanDataset(CORNAL_TEST, get_transform(train=False))
    sagit_dataset=PennFudanDataset(SAGIT_TEST, get_transform(train=False))

    axial_slices = axial_dataset.imgs_slices
    cornal_slices = cornal_dataset.imgs_slices
    sagit_slices = sagit_dataset.imgs_slices
    #初始化模型
    num_classes = NUM_CLASS
    axial_model = get_model_instance_segmentation(num_classes)
    axial_model.load_state_dict(torch.load(AXIAL_MODEL,map_location=torch.device('cpu')))
    axial_model.to(device)

    cornal_model = get_model_instance_segmentation(num_classes)
    cornal_model.load_state_dict(torch.load(CORNAL_MODEL, map_location=torch.device('cpu')))
    cornal_model.to(device)

    sagit_model = get_model_instance_segmentation(num_classes)
    sagit_model.load_state_dict(torch.load(SAGIT_MODEL, map_location=torch.device('cpu')))
    sagit_model.to(device)

'''
function: 检测切片是否有正交的boxes
'''
def Detect(model,data,indice):
    box_centers,boxes_pre=[],[]
    try:
        img, label,_ = data
        model.eval()
        with torch.no_grad():
            prediction=model([img.to(device)])
            boxes_pre = prediction[0]['boxes']
            # 合并重叠的区域
            boxes_pre = inter_rec(boxes_pre)
            if LOG:print("====>非axial截面检测到的区域{}".format(boxes_pre))
            # 求标注框的中心
            box_centers = [box_center(i) for i in boxes_pre]

            # TODO:如果需要显示单个的cornal或者sagit截面的检测情况，只需要取消下边所有行的注释
            # image=Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy())
            # #显示gd
            # boxes=list(np.array(label['boxes']))        #增加边框显示
            # ims_np = np.array(label['masks'], dtype="uint16")
            # mask = np.zeros(ims_np[0].shape)
            # for i in range(len(ims_np)):
            #     mask += ims_np[i] * (i + 1)  # 为了用不同的颜色显示出来
            # mask = Image.fromarray(mask).convert('L')
            # #显示图像
            # pres = np.zeros(prediction[0]['masks'].shape[1:])
            # print("类别数目:{}".format(len(prediction[0]["masks"])))
            # for i in range(prediction[0]['masks'].shape[0]):
            #     # print(np.unique(np.array(np.ceil(prediction[0]['masks'][i, 0]),dtype='int16')))
            #     pres += prediction[0]['masks'][i, 0].mul(
            #         (i + 1) * 255 // prediction[0]['masks'].shape[0]).byte().cpu().numpy()
            # pre = Image.fromarray(pres[0])
            # fig = plt.figure()
            # ax = fig.add_subplot(2, 2, 1)
            # plt.title('image')
            # for i in boxes:
            #     x0, y0, x1, y1 = i
            #     color = (random(), random(), random())
            #     rect = plt.Rectangle((x0, y0), abs(x1 - x0), abs(y1 - y0), edgecolor=color, fill=False, linewidth=1)
            #     ax.add_patch(rect)
            # plt.imshow(image)
            # plt.subplot(2, 2, 2)
            # plt.title('mask')
            # plt.imshow(mask)
            # # 显示预测的框在原图上的图像
            # ax2 = fig.add_subplot(2, 2, 3)
            # plt.title('image+pre_boxes')
            # rects = []
            # for i in boxes_pre:
            #     x0, y0, x1, y1 = i
            #     color = (random(), random(), random())
            #     rect = plt.Rectangle((x0, y0), abs(x1 - x0), abs(y1 - y0), edgecolor=color, fill=False, linewidth=1)
            #     rects.append([(x0, y0), abs(x1 - x0), abs(y1 - y0), color, False, 1])
            #     ax2.add_patch(rect)
            # plt.imshow(image)
            # # 显示预测的mask
            # ax3 = fig.add_subplot(2, 2, 4)
            # plt.title('prection')
            # for rect in rects:
            #     rect = plt.Rectangle(rect[0], rect[1], rect[2], edgecolor=rect[3], fill=rect[4], linewidth=rect[5])
            #     ax3.add_patch(rect)
            # plt.imshow(pre)
    except:
        if DEBUG_LOG:print("=======>sagit 或者 cornal 切片检测出错<==========")

    #判断是否正交，同时检测到,需要注意的是，显示的时候坐标x,y是反的
    #这里如果只判断中心过于严格，应该判断在一个区间内即可
    #if((indice[2],indice[1]) in box_centers):
    #判断中心坐标是否在box中
    for box in boxes_pre:
        if LOG:print("====>判断范围{}----------{}".format((indice[2],indice[1]),box))
        if (indice[2]>=box[0] and indice[2]<=box[2]) and (indice[1]>=box[1] and indice[1]<=box[3]):
            if LOG:print("*********匹配成功：{}---{}***********".format(indice,box))
            return True
    else:
        return False
"""
function: 根据检测框的中心坐标，去另外两个正交的截面判断是否检测出来
args: 输入的是一个检测box([x0,y0,x1,y1])和当前切片的位置indice
"""
def judge_cor_sig(box,indice):
    x=int(abs(box[0]-box[2])//2+box[0])
    y=int(abs(box[1]-box[3])//2+box[1])
    z=int(indice)
    if LOG:print(x,y,z)
    #这里获得另外截面的坐标,cor (1,2,0)
    coronal_coordinate=(y,x,z)
    sagit_coordinate=(x,y,z)

    coronal_contain,sagit_contain=False,False
    #做判断,coronal 中检测到的box是否包含（x,z）,sigit中是否包含（y,z）
    if str(coronal_coordinate[0]) in cornal_slices:
        coronal_contain=Detect(model=cornal_model,data=cornal_dataset[cornal_slices.index(str(coronal_coordinate[0]))],indice=coronal_coordinate)
    if not coronal_contain: return False
    if str(sagit_coordinate[0]) in sagit_slices:
        sagit_contain=Detect(model=sagit_model,data=sagit_dataset[sagit_slices.index(str(sagit_coordinate[0]))],indice=sagit_coordinate)
    if coronal_contain and sagit_contain:
        if LOG:print("\n*********骨折坐标{}************\n".format((x,y,z)))
        return True
    return False
"""
function: 用来正交判断在
args: 输入一个axial的切片的位置
"""
def decision(order):
    global EXIST
    EXIST=False
    img, label ,index= axial_dataset[order]
    model=axial_model
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        # 显示gd
        boxes = list(np.array(label['boxes']))  # 增加边框显示
        ims_np = np.array(label['masks'], dtype="uint16")
        mask = np.zeros(ims_np[0].shape)
        for i in range(len(ims_np)):
            mask += ims_np[i] * (i + 1)  # 为了用不同的颜色显示出来
        mask = Image.fromarray(mask).convert('L')
        boxes_pre = prediction[0]['boxes']
        # 合并重叠的区域
        boxes_pre = inter_rec(boxes_pre)
        # for i in boxes_pre:
        #     print(box_center(i))
        #判断三个正交的截面
        boxes_center = [box_center(i) for i in boxes_pre]
        if LOG:print("====>axial截面检测到的boxes中心:/t{}".format(boxes_center))
        boxes_pre=[list(i) for i in boxes_pre if judge_cor_sig(i,index)]
        if LOG:print("\n正交判断的结果:/t{}\n".format(boxes_pre))

        if boxes_pre:
            EXIST=True
            pres = np.zeros(prediction[0]['masks'].shape[1:])
            #print("类别数目:{}".format(len(prediction[0]["masks"])))
            for i in range(prediction[0]['masks'].shape[0]):
                # print(np.unique(np.array(np.ceil(prediction[0]['masks'][i, 0]),dtype='int16')))
                pres += prediction[0]['masks'][i, 0].mul(
                    (i + 1) * 255 // prediction[0]['masks'].shape[0]).byte().cpu().numpy()
            pre = Image.fromarray(pres[0])
            fig = plt.figure()
            ax = fig.add_subplot(2, 2, 1)
            plt.title('image')
            for i in boxes:
                x0, y0, x1, y1 = i
                color = (random(), random(), random())
                rect = plt.Rectangle((x0, y0), abs(x1 - x0), abs(y1 - y0), edgecolor=color, fill=False, linewidth=1)
                ax.add_patch(rect)
            plt.imshow(image)
            plt.subplot(2, 2, 2)
            plt.title('mask')
            plt.imshow(mask)
            # 显示预测的框在原图上的图像
            ax2 = fig.add_subplot(2, 2, 3)
            plt.title('image+pre_boxes')
            rects = []
            for i in boxes_pre:
                x0, y0, x1, y1 = i
                color = (random(), random(), random())
                rect = plt.Rectangle((x0, y0), abs(x1 - x0), abs(y1 - y0), edgecolor=color, fill=False, linewidth=1)
                rects.append([(x0, y0), abs(x1 - x0), abs(y1 - y0), color, False, 1])
                ax2.add_patch(rect)
            plt.imshow(image)
            # 显示预测的mask
            ax3 = fig.add_subplot(2, 2, 4)
            plt.title('prection')
            for rect in rects:
                rect = plt.Rectangle(rect[0], rect[1], rect[2], edgecolor=rect[3], fill=rect[4], linewidth=rect[5])
                ax3.add_patch(rect)
            plt.imshow(pre)

"""
function: 评估单个截面
"""
def evalutation(model_name,datapath):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 1+4

    # use our dataset and defined transformations
    dataset = PennFudanDataset(datapath, get_transform(train=True))
    dataset_test = PennFudanDataset(datapath, get_transform(train=False))

    # split the dataset in train and test set
    #indices = torch.randperm(len(dataset)).tolist()
    indices=[i for i in range(len(dataset))]
    dataset = torch.utils.data.Subset(dataset, indices[:])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(os.path.join('./Weights',model_name),map_location=torch.device('cpu')))
    # move model to the right device
    model.to(device)

    #example
    img,label=dataset_test[2]
    model.eval()
    with torch.no_grad():
        prediction=model([img.to(device)])
        image=Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy())
        #显示gd
        boxes=list(np.array(label['boxes']))        #增加边框显示
        ims_np = np.array(label['masks'], dtype="uint16")
        mask = np.zeros(ims_np[0].shape)
        for i in range(len(ims_np)):
            mask += ims_np[i] * (i + 1)  # 为了用不同的颜色显示出来
        mask = Image.fromarray(mask).convert('L')

        boxes_pre=prediction[0]['boxes']

        #合并重叠的区域
        boxes_pre=inter_rec(boxes_pre)
        pres=np.zeros(prediction[0]['masks'].shape[1:])
        #print("类别数目:{}".format(len(prediction[0]["masks"])))
        for i in range(prediction[0]['masks'].shape[0]):
            #print(np.unique(np.array(np.ceil(prediction[0]['masks'][i, 0]),dtype='int16')))
            pres+=prediction[0]['masks'][i, 0].mul((i+1)*255//prediction[0]['masks'].shape[0]).byte().cpu().numpy()
        pre = Image.fromarray(pres[0])
        fig=plt.figure()
        ax=fig.add_subplot(2,2,1)
        plt.title('image')
        for i in boxes:
            x0,y0,x1,y1=i
            color=(random(),random(),random())
            rect=plt.Rectangle((x0,y0),abs(x1-x0),abs(y1-y0),edgecolor=color,fill=False,linewidth=1)
            ax.add_patch(rect)
        plt.imshow(image)
        plt.subplot(2,2,2)
        plt.title('mask')
        plt.imshow(mask)
        #显示预测的框在原图上的图像
        ax2 = fig.add_subplot(2, 2, 3)
        plt.title('image+pre_boxes')
        rects=[]
        for i in boxes_pre:
            x0, y0, x1, y1 = i
            color = (random(), random(), random())
            rect = plt.Rectangle((x0, y0), abs(x1 - x0), abs(y1 - y0), edgecolor=color, fill=False, linewidth=1)
            rects.append([(x0, y0), abs(x1 - x0), abs(y1 - y0), color, False, 1])
            ax2.add_patch(rect)
        plt.imshow(image)
        #显示预测的mask
        ax3=fig.add_subplot(2,2,4)
        plt.title('prection')
        for rect in rects:
            rect=plt.Rectangle(rect[0],rect[1],rect[2],edgecolor=rect[3],fill=rect[4],linewidth=rect[5])
            ax3.add_patch(rect)
        plt.imshow(pre)
        #plt.show()

if __name__=='__main__':
    #评估单个切片
    evalutation("axial.pt","/Users/jinxiaoqiang/jinxiaoqiang/ModelsGenesis/pytorch/axial_test_slice")
    # evalutation("sagit.pt","/Users/jinxiaoqiang/jinxiaoqiang/ModelsGenesis/pytorch/sagital_test_slice")
    # evalutation("cornal.pt","/Users/jinxiaoqiang/jinxiaoqiang/ModelsGenesis/pytorch/coronal_test_slice")

    #评估存在label的数据，正交判断
    # Init()
    # N=len(axial_dataset)
    # print("切片个数为：{}".format(N))
    # #需要判断第几个数据
    # for order in range(N):
    #     try:
    #         decision(order)
    #         if EXIST:
    #             print("第{}张切片".format(order+1))
    #     except:
    #         print("第{}张没有检测到骨折部分".format(order+1))

    # 判断单个nii.gz文件，并且生成加mask的nii.gz文件
    # Init()
    # nii_path=NII_GZ
    # nii_savepath=NII_GZ_SAVE
    # if not os.path.isdir(nii_savepath):
    #     os.makedirs(join(nii_savepath,'image'))
    #     os.makedirs(join(nii_savepath,'mask'))
    # for i in os.listdir(nii_path):
    #     print("\n\n{}\n\n".format(i))
    #     nii=join(nii_path,i)
    #     #nii="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/ribfrac-val-images/RibFrac489-image.nii.gz"
    #     signal_nii(nii, nii_savepath)
    #     print("\n\n\n\n处理完一个\n\n\n")
    #     #break
    # plt.show()