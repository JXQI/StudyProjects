from random import random
from PIL import ImageDraw,Image
from matplotlib import pyplot as plt
import numpy as np
from settings import ATTENTION_LOG

#全局变量
QUENE=[]  #维护的队列
N_SLICES=3  #需要判别的上下层数目
output,mask_output=[],[] #保存nii数据和mask数据

"""
function: 判断是否有相交的box
args: 输入一个box和一个box_pre列表,其中box_pre是按照第一个元素升序排列的
return: true or false
"""
def judge(box,boxes_pre):
    status=False #是否有相交的boxes
    x0,y0,x1,y1=box
    for i,item in enumerate(boxes_pre):
        if ATTENTION_LOG:print("需要判断的boxes{}--{}".format(box,item))
        m0,n0,m1,n1=item[0]
        if not (m1<x0 or m0>x1 or n1<y0 or n0>y1):  #不相交，分别位于左右上下,点的坐标是左下点，右上角
            status=True
            boxes_pre[i][1]+=1  #为相交的box相交次数+1
            if ATTENTION_LOG:print("需要判断的boxes{}--{}相交".format(box,item[0]))
        else:
            if ATTENTION_LOG:print("不相交")
    if ATTENTION_LOG:print("判断后的box_pre{}".format(boxes_pre))
    return boxes_pre,status
"""
function: 清空队列
args: 队列,队列数据类型 {nii_data:np.array(),boxes_pre:[(box1,相交的次数),(box2,相交的次数)]
return: None
"""
def clear_quene(quene):
    global output
    global mask_output #保存nii数据和mask数据

    if ATTENTION_LOG:print("需要清空的队列{}".format(quene))
    while(len(quene)>0):
        data=quene.pop(0)
        image_gray=data["nii_data"]
        mask_gray=Image.new('L',size=image_gray.size,color=0)
        boxes_pre=data["boxes_pre"]
        for item in boxes_pre:
            i,j=item[0],item[1]
            if ATTENTION_LOG:print("相交的次数{}/{}".format(j,N_SLICES))
            if j>=N_SLICES:  #当检测框相交次数超过N的时候才画框保留，否则不画框
                x0, y0, x1, y1 = i
                # 在原图上画矩形框并且保存
                print("====>在原图上标记检测框:{}".format(i))
                draw = ImageDraw.Draw(image_gray)
                draw.rectangle(i, fill=None, outline="black")
                # 在mask上画矩形框并且保存
                print("====>在mask上标记检测框:{}".format(i))
                # 为mask着色
                mask_gray = np.array(mask_gray)
                mask_gray[int(x0):int(x1), int(y0):int(y1)] = boxes_pre.index(item) + 1
                print("\n标记位置{}\n".format(i))
        output.append(np.array(image_gray))
        mask_output.append(np.array(mask_gray))
    if ATTENTION_LOG:print("队列已经清空{}".format(QUENE))
"""
function: 维护一个有相交box的队列
args: 输入image_array(image是一个Image对象)和boxes_pre列表
return： 返回一个队列，队列数据类型 {nii_data:np.array(),boxes_pre:[(box1,相交的次数),(box2,相交的次数)]
"""
def attention(image_array,boxes_pre,number):
    global QUENE
    global output
    global mask_output

    if boxes_pre:  #是否检测到有交集的boxes，如果没有直接清空队列，直接返回
        # 将数据转化为队列的数据类型
        N = [0] * len(boxes_pre)
        data = {"nii_data": image_array, "boxes_pre": [list(i) for i in zip(boxes_pre, N)]}  #元组不能修改，这里需要转换成list
        if ATTENTION_LOG:print("需要检测的boxes{}".format(boxes_pre))
        if QUENE:
            if ATTENTION_LOG:print("队列不为空{}".format(QUENE))
            #判断是否于队列中的boxes有相交
            status=False
            for i in QUENE:  #更新了所有的boxes相交次数
                for j in i["boxes_pre"]:
                    res,res_status=judge(j[0],data["boxes_pre"])#如果相交返回True   #TODO:注意这里相交的话对当前的box数量也得加1
                    data["boxes_pre"] = res  # 更新新添加的boxes相交次数
                    if res_status:
                        j[1]+=1     #相交的次数+1
                        status=True #有相交的boxes
                    else: #不相交判断当前box次数是否大于N,否则移除
                        if j[1]<N_SLICES:
                            i["boxes_pre"].remove(j)
                        if ATTENTION_LOG:print("boxes判定结果不相交")
            # if status: #如果相交,改变一直接后边直接入队列即可
                #QUENE.append(data)
            if not status:#如果不相交，需要清空队列，保存新的队列
                if ATTENTION_LOG:print("没有相交的boxes")
                clear_quene(QUENE)
        QUENE.append(data)
        if ATTENTION_LOG:print("队列添加新的队列{}".format(QUENE))
    else:
        if ATTENTION_LOG:print("没有检测到boxes")
        #清空队列中的元素
        clear_quene(QUENE)
        #添加当前元素进输入列表
        output.append(np.array(image_array))
        mask_gray = Image.new('L', size=image_array.size, color=0)
        mask_output.append(np.array(mask_gray))
    print("切片队列大小{}/{}".format(len(output),number))

if __name__=='__main__':
   pass