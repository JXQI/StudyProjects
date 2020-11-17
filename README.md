# DTI_AD
数据下载：https://pan.cstcloud.cn/s/xHENmy4SQ14  
生成requirements.txt文件：pipreqs ./ --encoding=utf8

安装需要用到的库：  
pip install -r requirements.txt  
   
执行程序说明:  
1.首先需要对原始数据MCAD_AFQ_competition.mat进行读取操作，我们这里采取的方式是：对于纤维束的特征单独存储在每个.csv文件中，对于age、sex等特征存储在同一个文件中  
执行方式：python data_load.py  
执行结果：生成data文件夹，而且下边有700个num.csv的文件和一个data_result.csv文件  
  
2.接着进行测试集和验证集的划分：  
执行方式：python data_divide.py  其中默认参数 测试集：验证集=8：2 ratio=[0.8,0.2]  
执行结果：当前路径下生成train.txt和val.txt两个文件  

3.对原始数据进行处理，并且生成新的文件：0_0.csv等文件  
执行方式：python data_deal.py  
执行结果：./data路径下生成对应的0_0.csv文件  

4.运行：  
执行方式： ./run.sh  

5.测试方法：  
修改test.py或者test_three.py的权重路径，生成文件名  
对于二分类：python test.py  
对于三分类：python test_three.py  


TODO:  
1.利用机器学习的方法去做分类  
2.新的数据预处理方法  


