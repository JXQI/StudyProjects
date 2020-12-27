"""
function : 将png格式标注的数据集转换为coco格式
cite : https://patrickwasp.com/create-your-own-coco-style-dataset/
       https://programtip.com/zh/art-59220
"""

import json
import datetime
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from setting import SAVE_PATH
import shutil

INFO = {
    "description": "rib fracture",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "jinxiaoqiang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]
CATEGORIES = [
    {
        'id': 1,
        'name':'Displaced',
        'supercategory': 'rib fracture',
    },
    {
        'id': 2,
        'name': 'Nondisplaced',
        'supercategory': 'rib fracture',
    },
    {
        'id': 3,
        'name': 'Buckle',
        'supercategory': 'rib fracture',
    },
    {
        'id': 4,
        'name': 'Segmental',
        'supercategory': 'rib fracture',
    },
    {
        'id': 5,
        'name': 'Ignore',
        'supercategory': 'shape',
    },
]
def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg', '*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files

# 数据保存路径
data_dir='./data'
train_dir='./data/coco/train2017'
val_dir="./data/coco/val2017"
test_dir="./data/coco/test2017"
annotations_dir='./data/coco/annotations'
mask_dir='./data/coco/mask'
if not data_dir:
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    # os.makedirs(test_dir)  #直接后边生成val的软连接
    os.makedirs(mask_dir)

"""
fucntion : 输入需要生成annotations的image_path 和 对应的mask_path
"""
def main(image_dir,mask_dir):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    segmentation_id=1
    image_id=1
    for root, _, files in os.walk(image_dir):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            # image_id=os.path.basename(image_filename).split('.')[0]
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)
            print(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(mask_dir):
                annotation_files = filter_for_annotations(root, files, image_filename)
                print(annotation_files)
                # go through each associated annotation
                for annotation_filename in annotation_files:

                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                                             .convert('1')).astype(np.uint8)

                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
    if "train" in image_dir:
        annotations_name="instances_train2017.json"
    if "val" in image_dir:
        annotations_name="instances_val2017.json"
    annotations_name=os.path.join("./data/coco/annotations",annotations_name)
    with open(annotations_name, 'w') as f:
        json.dump(coco_output,f,indent=4)

if __name__=='__main__':
    # train
    main(train_dir,mask_dir)
    # val
    main(val_dir,mask_dir)
    # test 创建软连接，将test指向val,保证程序通过
    os.symlink("val2017",test_dir)
    os.symlink("instances_val2017.json","data/coco/annotations/instances_test2017.json")
    # 将生成的数据集移动到需要保存的位置
    shutil.move("data",SAVE_PATH)