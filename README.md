# DTI_AD
数据下载：https://pan.cstcloud.cn/s/xHENmy4SQ14  
生成requirements.txt文件：pipreqs ./ --encoding=utf8
   
执行程序说明:  
1.首先需要对原始数据MCAD_AFQ_competition.mat进行读取操作，我们这里采取的方式是：对于纤维束的特征单独存储在每个.csv文件中，对于age、sex等特征存储在同一个文件中  
执行方式：python data_load.py  
执行结果：生成data文件夹，而且下边有700个num.csv的文件和一个data_result.csv文件  
  
2.接着进行测试集和验证集的划分：  
执行方式：python data_divide.py  其中默认参数 测试集：验证集=8：2 ratio=[0.8,0.2]  
执行结果：当前路径下生成train.txt和val.txt两个文件  

3.运行：  
执行方式：python train.py

