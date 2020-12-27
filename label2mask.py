import numpy as np
from PIL import Image
import os
from os.path import join
from skimage.measure import label,regionprops
'''
function : 将label保存为多个二进制的mask
note : 这种方法有一个问题，只能适用于同一个图像中同一个标签只能出现一次，如果出现多个不能分成多个实例

'''
def label2mask(filename):
    label=np.array(Image.open(filename))
    obj_ids=np.unique(label)
    obj_ids = obj_ids[1:]
    # split the color-encoded mask into a set
    # of binary masks
    masks = label == obj_ids[:, None, None]

    return masks,obj_ids
"""
function : 将label中不同的实例保存为多个mask文件
"""
def label2mask_(filename):
    label_arr = np.array(Image.open(filename))
    # 对图像进行连通域操作，并对同一个连通区域的编号
    pre_label = label(label_arr > 0)
    # 去除背景的编号，只保留实例的
    pre_nums = np.unique(pre_label)[1:]
    # 对每个实例编码为二进制图像
    masks = pre_label == pre_nums[:, None, None]
    # 求每个连通区域的中心坐标，确定对应实例的类别信息
    pred_regions = regionprops(pre_label, label_arr)
    # 根据像素值确定类别信息
    pre_index = [region.max_intensity for region in pred_regions]

    return masks,pre_index

'''
function : 类别字典
'''
class_dictory={"1":'Displaced',"2":'Nondisplaced','3':'Buckle','4':'Segmental','5':'Ignore'}


if __name__=='__main__':
    path='./axial_slice/mask'
    for i in os.listdir(path):
        filename=join(path,i)
        savepath='annotations'
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        # filename='./axial_slice/mask/RibFrac9_148.png'
        masks,obj_ids=label2mask_(filename)
        for i in range(masks.shape[0]):
            class_name=class_dictory[str(obj_ids[i])]
            mask_name=os.path.basename(filename).split('.')[0]+'_'+class_name+'_'+str(i)+".png"
            mask=Image.fromarray(masks[i]).convert('1')
            mask.save(join(savepath,mask_name))
        # break

