import os
from os.path import join
import shutil
import random


"""
fucntion: 
    将X光安检图像识别2021挑战赛数据集转到VOC2007文件夹下
args: 
    datapath: x光数据集
    vocpath: voc文件夹
"""
def convert2Voc(datapath,vocpath):
    trainSet=join(datapath,"科大讯飞股份有限公司_X光安检图像识别2021挑战赛初赛第一阶段数据集","初赛第一阶段_训练集")
    testSet=join(datapath,"科大讯飞股份有限公司_X光安检图像识别2021挑战赛初赛第一阶段数据集","初赛第一阶段_测试集")

    VOC2007=join(vocpath,"VOCdevkit","VOC2007")
    # 将图像移动到JPEGImages下,标注文件移动到Annotations下
    for domain in os.listdir(trainSet):
        if "domain" in domain:
            domain_path=join(trainSet,domain)
            for item in os.listdir(domain_path):
                if item.endswith('.jpg'):
                    image=join(domain_path,item)
                    img_name=os.path.basename(item)[5:] # 保存图像的名称
                    shutil.move(image,join(VOC2007,"JPEGImages",img_name))
                elif item=='XML':
                    xml_path=join(domain_path,item)
                    for xml in os.listdir(xml_path):
                        if xml.endswith(".xml"):
                            xml=join(xml_path,xml)
                            xml_name=os.path.basename(xml)[5:]  # 长度小于9
                            shutil.move(xml,join(VOC2007,"Annotations",xml_name))

    # 移动测试集,注意测试集没有标注XML
    for item in os.listdir(testSet):
        if item.endswith('.jpg'):
            image = join(testSet, item)
            img_name = os.path.basename(item)  # 保存图像的名称
            shutil.move(image, join(VOC2007, "JPEGImages", img_name))
        elif item == 'XML':
            xml_path = join(testSet, item)
            for xml in os.listdir(xml_path):
                if xml.endswith(".xml"):
                    xml = join(xml_path, xml)
                    xml_name = os.path.basename(xml)
                    shutil.move(xml, join(VOC2007, "Annotations", xml_name))
    # 统计图片数目
    image=len([i for i in os.listdir(join(VOC2007,"JPEGImages")) if i.endswith('.jpg')])
    xml=len([i for i in os.listdir(join(VOC2007,"Annotations")) if i.endswith('.xml')])

    print("image_num:{},xml_num={}".format(image,xml))

"""
function: 
    划分数据集
args:
    path:VOCdevkit所在目录
    ratio: train:val 的比例
return:
    生成Main文件夹下包含了每个分类的train.txt、val.txt,test.txt和trainval.txt
"""

def split_dataSet(path,ratio=0.8):
    train_image=[i for i in os.listdir(join(path,'VOCdevkit','VOC2007','JPEGImages')) if i.endswith('.jpg') and 'test' not in i]
    test_image=[i for i in os.listdir(join(path,'VOCdevkit','VOC2007','JPEGImages')) if i.endswith('.jpg') and 'test' in i]
    print(len(train_image),len(test_image))
    train_precent=ratio

    train_list=range(len(train_image))
    train_set_len=int(len(train_image)*train_precent)
    train_set=random.sample(train_list,train_set_len)

    ftrain = open(join(path,'VOCdevkit',"VOC2007","ImageSets","Main","train.txt"),'w')
    fval = open(join(path,'VOCdevkit', "VOC2007", "ImageSets", "Main", "val.txt"), 'w')
    ftrainval = open(join(path,'VOCdevkit', "VOC2007", "ImageSets", "Main", "trainval.txt"), 'w')
    ftest = open(join(path,'VOCdevkit', "VOC2007", "ImageSets", "Main", "test.txt"), 'w')

    # 写入训练集
    for i in train_list:
        image_name=train_image[i].split('.')[0]+'\n'
        ftrainval.write(image_name)
        if i in train_set:
            ftrain.write(image_name)
        else:
            fval.write(image_name)
    # 写入测试集
    # 排序，使图片有顺序，从小到大
    test_image=sorted(test_image)
    print(test_image)
    for image in test_image:
        image_name=image.split('.')[0]+'\n'
        ftest.write(image_name)

    # 关闭文件
    ftrain.close()
    fval.close()
    ftrainval.close()
    ftest.close()

"""
function: 
    得到类别信息
args: 
    xml 文件夹路径
"""
def get_classes(xml_path):
    pass


if __name__=="__main__":
    datapath='/home/victoria/train_data_jinxiaoqiang/data_set'
    vocpath="/home/victoria/train_data_jinxiaoqiang/data_set/data"
    # 创建VOC文件夹
    convert2Voc(datapath,vocpath)
    # 数据集划分
    split_dataSet(vocpath)