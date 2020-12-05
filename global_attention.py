from random import random
from PIL import ImageDraw,Image
from matplotlib import pyplot as plt
import numpy as np

#全局变量
QUENE=[]  #维护的队列
N=3  #需要判别的上下层数目
ouput,mask_output=[],[] #保存nii数据和mask数据

"""
function: 判断是否有相交的box
args: 输入一个box和一个box_pre列表,其中box_pre是按照第一个元素升序排列的
return: true or false
"""
def judge(box,boxes_pre):
    x0,y0,x1,y1=box
    for item in boxes_pre:
        m0,n0,m1,n1=item
        if (m1<x0 or m0>x1 or n1>y0 or n0<y1):  #不相交，分别位于左右上下
            return False
    return True

"""
function: 清空队列
args: 队列,队列数据类型 {nii_data:np.array(),boxes_pre:[(box1,相交的次数),(box2,相交的次数)]
return: None
"""
def clear_quene(quene):
    global ouput
    global mask_output #保存nii数据和mask数据
    while(len(quene)>0):
        data=quene.pop(0)
        image_gray=data["nii_data"]
        mask_gray=Image.new('L',size=image_gray.size,color=0)
        boxes_pre=data["boxes_pre"]
        for item in boxes_pre:
            i,j=item[0],item[1]
            if j>=N:  #当检测框相交次数超过N的时候才画框保留，否则不画框
                x0, y0, x1, y1 = i
                # 在原图上画矩形框并且保存
                print("====>在原图上标记检测框:{}".format(i))
                draw = ImageDraw.Draw(image_gray)
                draw.rectangle(i, fill=None, outline="black")
                # 在mask上画矩形框并且保存
                print("====>在mask上标记检测框:{}".format(i))
                # 为mask着色
                mask_gray = np.array(mask_gray)
                mask_gray[int(x0):int(x1), int(y0):int(y1)] = boxes_pre.index(i) + 1
        ouput.append(image_gray)
        mask_output.append(mask_gray)
"""
function: 维护一个有相交box的队列
args: 输入image_array(image是一个Image对象)和boxes_pre列表
return： 返回一个队列，队列数据类型 {nii_data:np.array(),boxes_pre:[(box1,相交的次数),(box2,相交的次数)]
"""
def attention(image_array,boxes_pre):
    global QUENE

    if boxes_pre:  #是否检测到有交集的boxes，如果没有直接清空队列，直接返回
        # 将数据转化为队列的数据类型
        number = [0] * len(boxes_pre)
        data = {"nii_data": image_array, "boxes_pre": zip(boxes_pre, number)}
        if QUENE:
            #判断是否于队列中的boxes有相交
            status=False
            for i in QUENE:  #更新了所有的boxes相交次数
                for j in i["boxes_pre"]:
                    if judge(j[0],boxes_pre):#如果相交返回True
                        j[1]+=1     #相交的次数+1
                        status=True #有相交的boxes
                    else: #不相交判断当前box次数是否大于N,否则移除
                        if j[1]<N:
                            i.remove(j)
            if status: #如果相交直接入队列即可
                QUENE.append(data)
            else:#如果不相交，需要清空队列，保存新的队列
                clear_quene(QUENE)
                QUENE.append(data)
        else:
            QUENE.append(data)
    else:
        #清空队列中的元素
        clear_quene(QUENE)
        #添加当前元素进输入列表
        ouput.append(image_array)
        mask_gray = Image.new('L', size=image_array.size, color=0)
        mask_output.append(mask_gray)

if __name__=='__main__':
   pass