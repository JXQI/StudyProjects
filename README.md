# DTI_AD  
任务描述：  
弥散磁共振影像（DTI）在阿尔茨海默病（Alzheimer's disease, AD）中应用广泛，从DTI影像中提取扩散参数可以用来描述白质结构的完整性，进而显示AD中脑白质的退化模式。前期大量研究表明，使用基于DTI的白质指标，利用机器学习的方法可以比较有效的对AD进行诊断和分类。但是以往的绝大部分研究是基于单中心的有放回的交叉验证方法来评估分类的有效性，分类特征和方法的泛化性能有待进一步验证。为此，本项目旨在以18条主要的脑白质纤维束的扩散指标作为特征，建立并评估出对AD和健康人群（normal controls，NC）进行分类的最优机器学习模型。我们的首要目标是在交叉验证中和不同站点的数据集中获得较高且较为稳定的精度，在此基础上，进一步探索白质扩散信号对于轻度认知损害患者（mild cognitive impairment, MCI）的预测性能，即在AD、NC、MCI三分类中的性能表现。  

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
3.交叉验证的方法选取最优模型  
  


