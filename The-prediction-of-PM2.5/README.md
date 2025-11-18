# The-prediction-of-PM2.5
## 李宏毅课程作业一:regression

### 作业要求:
    给定训练集train.csv，要求根据前9个小时的空气监测情况预测第10个小时的PM2.5含量
### 数据集:
    (1)CSV文件，包含台湾丰原地区240天的气象观测资料(取每个月前20天的数据做训练集，12月X20天=240天，每月后10天数据用于测试，对学生不可见);
    (2)每天的监测时间点为0时，1时......到23时，共24个时间节点;
    (3)每天的检测指标包括CO、NO、PM2.5、PM10等气体浓度，是否降雨、刮风等气象信息，共计18项；
    (4)数据集地址：https://pan.baidu.com/s/1o2Yx42dZBJZFZqCa5y3WzQ，提取码：qgtm
### 问题:
    1.输入是什么?输出是什么?
        连续九个小时的PM2.5,第十个小时的作为label,输入第十个小时的预测值
    2.数据如何预处理?
        RAINFALL这一选项如果下雨为1,否则为NR,这里将NR替换为0
    3.数据如何加载?model怎么选择?损失函数如何计算?
     数据通过列表，model选择最简单的线性模型，损失函数选择平方差  
### 实验结果：
    loss曲线：见loss.png，可以看到模型是收敛的
    loss:The loss of validation is 0.691848(在验证集上是这个值，感觉有问题，后边接触到再去改正)
        
### 参考资料：
https://www.cnblogs.com/HL-space/p/10676637.html 
李宏毅课程PPT和作业资料：
链接: https://pan.baidu.com/s/19zjCiCN1vYoBYjPzAEih4A 提取码: 3q7q 
