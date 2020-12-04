import os
import numpy as np
import torch
from PIL import Image


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "image"))))
        sort_index = [int(self.imgs[i].split('_')[1].split('.')[0]) for i in range(len(self.imgs))]
        self.imgs = [x for _, x in sorted(zip(sort_index, self.imgs))]
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.masks=[x for _, x in sorted(zip(sort_index, self.masks))]
        #添加切片位置的索引
        self.imgs_slices=[idx.split('_')[1].split('.')[0] for idx in self.imgs]

    def __getitem__(self, idx):
        try:
            # load images ad masks
            img_path = os.path.join(self.root, "image", self.imgs[idx])
            mask_path = os.path.join(self.root, "mask", self.masks[idx])
            img = Image.open(img_path).convert("RGB")
            # note that we haven't converted the mask to RGB,
            # because each color corresponds to a different instance
            # with 0 being background
            mask = Image.open(mask_path)
            # convert the PIL Image into a numpy array
            mask = np.array(mask)
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]
            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax+1, ymax+1])    #TODO

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                img, target = self.transforms(img, target)
            #axial切片的位置信息
            index=self.imgs_slices[idx]
            #print(img_path)
            return img, target,index
        except:
            pass

    def __len__(self):
        return len(self.imgs)