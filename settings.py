"""
设置需要的参数，方便后期修改
"""

#训练参数
GPU=False
EPOCH=5
BATCH_SIZE=8
NUM_CLASS=1+5

#数据集路径
IMAGE="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/axial_slice"

#保存的模型路径
# AXIAL_MODEL="./Weights/axial.pt"
AXIAL_MODEL="./Weights/axial_mulclass.pt"
CORNAL_MODEL="./Weights/cornal.pt"
SAGIT_MODEL="./Weights/sagit.pt"
#训练数据集的路径
AXIAL_TEST="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/axial_slice"
CORNAL_TEST="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/axial_slice"
SAGIT_TEST="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/axial_slice"

#测试集路径
NII_GZ="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/ribfrac-val-images/"
#生成的结果保存路径
NII_GZ_SAVE="./nii_test_mulclass_signalslice_threeview"
#生成用于测试的结果保存路径
NII_GA_PRE='./nii_pre_mulclass_signalslice_threeview'

#窗口
HU_WINDOW=1100
HU_LEVEL=750

#保存模型的名字
MODEL_NAME='axial_mulclass.pt'

#是否开启连续层直接的判断
ATTENTION=False

#判断日志打印(print)是否开启
LOG=True
ATTENTION_LOG=False
DEBUG_LOG=True
