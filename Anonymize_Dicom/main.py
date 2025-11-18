import os
from os.path import join
# import pydicom
# from pydicom.data import get_testdata_file
import dicom2nifti
from tqdm import tqdm
import argparse

"""
function:
    回调函数：匿名患者姓名
args:
    (dataset类回调)
return:
    (直接修改元素，无返回值)
"""
def person_names_callback(dataset, data_element):
    if data_element.VR == "PN":
        data_element.value = "anonymous"

"""
function:
    单个病人的dicom图像匿名化，并且转换成nii.gz文件
args:
    path : 该病人dicom所在的文件夹路径
    dstpath : 需要保存的位置
    data_elements : 需要修改的属性
"""
class anonymize_dicom:
    def __init__(self,path,dstpath=None,data_elements=None):
        self.path=path
        self.dstpath=dstpath
        self.data_elements=data_elements
        self.dicom_list=[join(self.path,i) for i in os.listdir(path) if i.endswith('.dcm')]

    def anonymize_elements(self,singal_dicom):
        """
        function:
            读取单张dicom文件，并且修改属性
        args:
            singal_dicom:单张dicom文件
        return:
            匿名后的dicom文件，并且转换成narry格式
        """
        dataset = pydicom.dcmread(singal_dicom)
        dataset.walk(person_names_callback)

    def get_nii_gz(self):
        for item in self.dicom_list:
            self.anonymize_elements(item)

"""
function: 
    使用dicom2nifiti库
args:
    path: dicom文件夹路径
    dstpath: 需要保存的文件夹
return:
    返回nii.gz格式文件
"""
def get_nii(path,dstpath):
    dicom_list=[join(path,i) for i in os.listdir(path) if os.path.isdir(join(path,i))]
    for i,item in tqdm(enumerate(dicom_list)):
        item_name=join(dstpath,str(i).zfill(3))
        dicom2nifti.dicom_series_to_nifti(item,item_name)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path', type=str,help='the path of dicom file')
    parser.add_argument('dst_path', type=str,help='the path to get .nii file')
    parser.add_argument('-help', type=str, help='convert dicom to nii,please input srcpath and dstpath')
    args = parser.parse_args()
    path=args.src_path
    dstpath=args.dst_path
    # path='/Users/jinxiaoqiang/Desktop/test/manifest-1629983911363/Head-Neck Cetuximab-Demo/0522c0001/08-23-1999-NeckHeadNeckPETCT-03251'
    # dstpath="/Users/jinxiaoqiang/Desktop"
    get_nii(path,dstpath)
