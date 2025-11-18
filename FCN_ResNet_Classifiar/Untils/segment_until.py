import torch
import torch.nn.functional as F
import numpy as np
from Loader.segment_loader import label2image
from matplotlib import pyplot as plt
from PIL import Image
from os.path import join

#显示预测的图像
def show(image,label,predict):
    label=label2image(label)
    predict=label2image(predict)
    image=np.array(image).transpose((1,2,0))
    plt.subplot(1,3,1)
    plt.title("image")
    plt.imshow(image)   #TODO:这里输出是(3,160,160),需要的是(160,160)
    plt.subplot(1,3,2)
    plt.title("label")
    plt.imshow(label,cmap='gray')
    plt.subplot(1,3,3)
    plt.title("predict")
    plt.imshow(predict,cmap='gray')
    plt.show()

def Accuracy(net,dataloader,loss_function,device,crop_size):
    loss_get=0
    loss=[]
    total=0
    correct=0
    net.eval()
    number_pixes=crop_size[0]*crop_size[1]
    iou=0
    with torch.no_grad():
        for i,data in enumerate(dataloader,0):
            inputs,labels,image_name=data[0].to(device),data[1].to(device),data[2]
            net=net.to(device)
            outputs=net(inputs)
            _,predicted=torch.max(outputs,1)
            #print(predicted)
            #debug 显示预测的图像和label
            #show(inputs[0],labels[0],predicted[0])
            ################
            iou+=iou_mean(predicted,labels)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            #print(predicted,labels)
            temp=loss_function(outputs,labels)
            loss.append(temp)
            loss_get+=temp
        return loss_get/total,correct/(total*number_pixes),loss,iou/total
'''
计算IOU
'''
def iou_mean(pred,target,n_classes=1):
    # n_classes ：the number of classes in your dataset,not including background
    # for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    #pred = torch.from_numpy(pred)
    pred = pred.view(-1)
    #target = np.array(target)
    #target = torch.from_numpy(target)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes + 1):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
    return iousSum / n_classes

