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

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
#indices = torch.randperm(len(dataset)).tolist()
indices=[i for i in range(len(dataset))]
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load(os.path.join('./Weights',"mask_r_cnn.pt"),map_location=torch.device('cpu')))
# move model to the right device
model.to(device)

#example
img,label=dataset_test[-1]
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
    for i in range(prediction[0]['masks'].shape[0]):
        pres+=prediction[0]['masks'][i, 0].mul((i+1)*255//prediction[0]['masks'].shape[0]).byte().cpu().numpy()
    pre = Image.fromarray(pres[0])
    fig=plt.figure()
    ax=fig.add_subplot(1,3,1)
    plt.title('image')
    for i in boxes:
        x0,y0,x1,y1=i
        color=(random(),random(),random())
        rect=plt.Rectangle((x0,y0),abs(x1-x0),abs(y1-y0),edgecolor=color,fill=False,linewidth=1)
        ax.add_patch(rect)
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.title('mask')
    plt.imshow(mask)
    ax3=fig.add_subplot(1,3,3)
    plt.title('prection')
    for i in boxes_pre:
        x0,y0,x1,y1=i
        color=(random(),random(),random())
        rect=plt.Rectangle((x0,y0),abs(x1-x0),abs(y1-y0),edgecolor=color,fill=False,linewidth=1)
        ax3.add_patch(rect)
    plt.imshow(pre)
    plt.show()