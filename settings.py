"""
设置需要的参数，方便后期修改
"""

#训练参数
GPU=False
EPOCH=5
BATCH_SIZE=8
NUM_CLASS=2

#数据集路径
IMAGE="/Users/jinxiaoqiang/jinxiaoqiang/ModelsGenesis/pytorch/axial_test_slice"
#窗口
HU_WINDOW=1000
HU_LEVEL=700

#保存模型的名字
MODEL_NAME='axial.pt'