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
dataset = torch.utils.data.Subset(dataset, indices[:-500])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-500:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# get the model using our helper function
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load(os.path.join('./Weights',"mask_r_cnn.pt")))
# move model to the right device
model.to(device)

#example
img,label=dataset_test[-1]
model.eval()
with torch.no_grad():
    prediction=model([img.to(device)])
    image=Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy())
    #显示gd
    ims_np = np.array(label['masks'], dtype="uint16")
    mask = np.zeros(ims_np[0].shape)
    print(mask.shape)
    for i in range(len(ims_np)):
        mask += ims_np[i] * (i + 1)  # 为了用不同的颜色显示出来
    mask = Image.fromarray(mask).convert('L')
    # temp=label['masks'].mul(255).permute(1,2,0).byte().numpy()
    # if temp.shape[2]>1:
    #     mask=Image.fromarray(temp)
    # else:
    #     mask=Image.fromarray(temp.reshape(temp.shape[:2]))
    #pre=Image.fromarray(prediction[0]['masks'][3,0].mul(255).byte().cpu().numpy())
    pres=np.zeros(prediction[0]['masks'].shape[1:])
    print(pres.shape)
    for i in range(prediction[0]['masks'].shape[0]):
        print(i)
        pres+=prediction[0]['masks'][i, 0].mul((i+1)*255//prediction[0]['masks'].shape[0]).byte().cpu().numpy()
    pre = Image.fromarray(pres[0])
    #pre = Image.fromarray(prediction[0]['masks'].sum(axis=0)[0].mul(255).cpu().numpy())
    plt.subplot(1,3,1)
    plt.title('image')
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.title('mask')
    plt.imshow(mask)
    plt.subplot(1,3,3)
    plt.title('prection')
    #pre=pre.convert('L')
    plt.imshow(pre)
    plt.imshow(pre,cmap='tab10')
    plt.show()