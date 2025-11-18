import os
from os.path import join


"""
function: 创建VOC2007格式的文件夹
args: 创建的路径
"""
def create_VOCDataSet(path):
    VOCdevkit="VOCdevkit"
    VOC="VOC2007"
    Annotations="Annotations" # 该文件下存放的是xml格式的标签文件，每个xml文件都对应于JPEGImages文件夹的一张图片
    JPEGImages="JPEGImages" # 改文件夹下存放的是数据集图片，包括训练和测试图片
    ImageSets="ImageSets" # 该文件夹下存放了三个文件，分别是Layout、Main、Segmentation。在这里我们只用存放图像数据的Main文件，其他两个暂且不管
    Layout="Layout" # Layout下存放的是具有人体部位的数据（人的head、hand、feet等等，这也是VOC challenge的一部分）
    Main="Main" # Main文件夹下包含了每个分类的train.txt、val.txt和trainval.txt，需要保证的是train和val两者没有交集
    Segmentation="Segmentation" # Segmentation下存放的是可用于分割的数据。

    SegmentationClass="SegmentationClass"
    SegmentationObject="SegmentationObject" # 这两个文件都是与图像分割相关，跟本文无关，暂且不管

    # 创建相应的文件夹
    if not os.path.exists(join(path,VOCdevkit,VOC,Annotations)):
        os.makedirs(join(path,VOCdevkit,VOC,Annotations))
    if not os.path.exists(join(path,VOCdevkit, VOC, JPEGImages)):
        os.makedirs(join(path,VOCdevkit, VOC, JPEGImages))
    if not os.path.exists(join(path,VOCdevkit, VOC, ImageSets)):
        os.makedirs(join(path,VOCdevkit, VOC, ImageSets))
    if not os.path.exists(join(path,VOCdevkit, VOC, SegmentationClass)):
        os.makedirs(join(path,VOCdevkit, VOC, SegmentationClass))
    if not os.path.exists(join(path,VOCdevkit, VOC, SegmentationObject)):
        os.makedirs(join(path,VOCdevkit, VOC, SegmentationObject))
    # 创建ImageSets的子文件夹
    if not os.path.exists(join(path,VOCdevkit, VOC,ImageSets,Layout)):
        os.makedirs(join(path,VOCdevkit, VOC, ImageSets,Layout))
    if not os.path.exists(join(path,VOCdevkit, VOC, ImageSets,Main)):
        os.makedirs(join(path,VOCdevkit, VOC, ImageSets, Main))
    if not os.path.exists(join(path,VOCdevkit, VOC, ImageSets,Segmentation)):
        os.makedirs(join(path,VOCdevkit, VOC, ImageSets, Segmentation))

if __name__=="__main__":
    path="/Users/jinxiaoqiang/jinxiaoqiang/detection_match/data"
    create_VOCDataSet(path)




