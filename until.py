import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from os.path import  join
import time
from sklearn.metrics import roc_curve,roc_auc_score,f1_score
import torch.nn.functional as F
import numpy as np


# net:trained model
# dataloader:dataloader class
#loss_function: loss choose
#device: gpu or cpu
def Accuracy(net,dataloader,loss_function,device):
    loss_get=0
    loss=[]
    total=0
    correct=0
    net.eval()
    label,target,predict=[],[],[]
    with torch.no_grad():
        for i,data in enumerate(dataloader,0):
            inputs=data[0].to(device)
            labels=data[1].to(device)
            net=net.to(device)
            outputs=net(inputs)
            _,predicted=torch.max(outputs,1)
            #将label和概率添加进列表中去
            for lp in range(len(labels)):
                label.append(int(labels[lp]))
                target.append(float(F.softmax(outputs[lp], dim=0)[1]))
                predict.append(predicted[lp])

            print(predicted,labels)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            #print(predicted,labels)
            temp=loss_function(outputs,labels)
            loss.append(temp)
            loss_get+=temp
        return loss_get/total,correct/total,loss,label,target,predict

def drawline(x,y,xlabel,ylabel,title):
    path='./result'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.savefig('./result/'+title+'.jpg')
    #plt.show()  # TODO :为了同时显示多个图，将这个移除到最后，这里其实可以改为一个类的


#存储结果为csv文件
class SaveCsv:
    def __init__(self,name,path,file_name=None):
        self.name=name
        if file_name in os.listdir(path):
            os.remove(join(path,file_name))
        df = pd.DataFrame(data=[name], columns=name)
        if not os.path.isdir(path):
            os.mkdir(path)
        if file_name:
            self.path=join(path,file_name)
        else:
            self.path=join(path,time.strftime("%Y-%m-%d", time.localtime(time.time()))+'-'+time.strftime("%H-%M-%S",time.localtime(time.time()))+"-result.csv")
        df.to_csv((self.path), encoding="utf-8-sig", mode="a", header=False, index=False)


    def savefile(self, my_list):
        """
        把文件存成csv格式的文件，header 写出列名，index写入行名称
        :param my_list: 要存储的一条列表数据
        :return:
        """
        df = pd.DataFrame(data=[my_list],columns=self.name)
        df.to_csv(self.path, encoding="utf-8-sig", mode="a", header=False, index=False)

    def saveAll(self):
        """
        一次性存储完
        :return:
        """
        pf = pd.DataFrame(data=self.clist)
        pf.to_csv(self.path, encoding="utf-8-sig", header=False, index=False)

class Draw_ROC:
    def __init__(self,path,label):
        self.label=label
        self.path=path
    #输入类别和预测结果
    def ROC(self,label,predict,name):
        label=np.array(label)
        predict=np.abs(predict)
        fpr, tpr, thresholds = roc_curve(label, predict)
        #计算分数
        score=self.roc_score(label,predict)
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=name+self.label+'_'+str(round(score,2)))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        #plt.show()
        plt.savefig(join(self.path, name+'_'+self.label + '_roc.jpg'))
    def roc_score(self,label,predict):
        label = np.array(label)
        predict = np.abs(predict)
        self.score=roc_auc_score(label,predict)
        return self.score
    def f1_score(self,label,predict):
        label = np.array(label)
        predict = np.abs(predict)
        score=f1_score(label,predict,average='micro')
        return score

if __name__ == '__main__':
    print("当前时间::" + time.strftime("%Y-%m-%d", time.localtime(time.time()))+'-'+time.strftime("%H-%M-%S",time.localtime(time.time())))
    name = ['Net', 'batch_size', 'lr', "is_drop","epoch", "accu"]
    sc = SaveCsv(name=name,path='./',file_name="hahahh.csv")
    #sc.main()
    #sc.saveAll()
    for i in range(5):
        list_name=[i,2,3,4,5,i]
        sc.savefile(my_list=list_name,name=name)
    # name=['id','uid','time']
    # df = pd.DataFrame(data=list, columns=name)
    # df.to_csv("./Result/result.csv", encoding="utf-8-sig", mode="a", header=True, index=False)