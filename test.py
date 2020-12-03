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
from settings import IMAGE,MODEL_NAME

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def evalutation(model_name,datapath):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
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
        pres=np.zeros(prediction[0]['masks'].shape[1:])
        print("类别数目:{}".format(len(prediction[0]["masks"])))
        for i in range(prediction[0]['masks'].shape[0]):
            #print(np.unique(np.array(np.ceil(prediction[0]['masks'][i, 0]),dtype='int16')))
            #TODO:
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
        for i in boxes_pre:
            x0, y0, x1, y1 = i
            color = (random(), random(), random())
            rect = plt.Rectangle((x0, y0), abs(x1 - x0), abs(y1 - y0), edgecolor=color, fill=False, linewidth=1)
            ax2.add_patch(rect)
        plt.imshow(image)
        #显示预测的mask
        ax3=fig.add_subplot(2,2,4)
        plt.title('prection')
        for i in boxes_pre:
            x0,y0,x1,y1=i
            color=(random(),random(),random())
            rect=plt.Rectangle((x0,y0),abs(x1-x0),abs(y1-y0),edgecolor=color,fill=False,linewidth=1)
            ax3.add_patch(rect)
        plt.imshow(pre)
        #plt.show()

if __name__=='__main__':
    evalutation("axial.pt","/Users/jinxiaoqiang/jinxiaoqiang/ModelsGenesis/pytorch/axial_test_slice")
    #evalutation("sagit.pt","/Users/jinxiaoqiang/jinxiaoqiang/ModelsGenesis/pytorch/sagital_test_slice")
    #evalutation("cornal.pt","/Users/jinxiaoqiang/jinxiaoqiang/ModelsGenesis/pytorch/coronal_test_slice")
    plt.show()