import pandas as pd
import numpy as np
from matplotlib import pyplot as plot

def DataProcess(df):
    train_x=[]
    labels=[]
    #df=df.replace(['NR'],[0.0]) #这两种方法都是可以替换掉的
    df = df.replace('NR', 0.0)
    array=np.array(df).astype(float)
    #将数据集拆分，只需要pm2.5的数据
    for i in range(0,array.shape[0],18):    #注意这里range()可以配置步长
        for j in range(array.shape[1]):
            if (j+10)<=array.shape[1]:
                mat=array[i][j:j+9]
                label=array[i][j+9]
                train_x.append(mat)
                labels.append(label)
    # for i in range(int(array.shape[0]/18)):    #array.shape[0]/18) 这里的目的主要是只挑选pm2.5的数据
    #     for j in range(array.shape[1]):
    #         if (j+10)<=array.shape[1]:
    #             {train_x.append(array[9+i*18][j+k]) for k in range(9)}
    #             labels.append(array[9+i*18][j+9])
    train_x=np.array(train_x)
    labels=np.array(labels)
    return train_x,labels
def train(x,labels,epoch):
    weight=np.zeros(9)  #weight
    bias=0      #bias
    biasG_Sum=0
    weigG_Sum=np.zeros(9)
    l_re = 0.001  # regulazation
    l_r = 0.1  # learning rate
    loss_arr=[]

    for e in range(epoch):
        biasG=0
        weigG=np.zeros(9)
        for i in range(len(labels)):
            #biasG+=(labels[i]-weight.dot(x[9*i:9*(i+1)])-bias)*(-2)
            biasG += (labels[i] - weight.dot(x[i]) - bias) * (-2)
            for j in range(9):
                #weigG[j]+=(labels[i]-weight.dot(x[9*i:9*(i+1)])-bias)*(-2*(x[9*i+j]))
                weigG[j] += (labels[i] - weight.dot(x[i]) - bias) * (-2 * (x[i][j]))
            #求平均
        biasG /= len(labels)
        weigG /= len(labels)
        #regularization 添加正则项，平缓作用
        for k in range(9):
            weigG[k]+=2*l_re*weight[k]
        #Adagrad 过往梯度的累计和
        biasG_Sum+=biasG**2
        weigG_Sum+=weigG**2
        bias=bias-l_r*biasG/(biasG_Sum**(0.5))
        weight=weight-l_r*weigG/(weigG_Sum**(0.5))

        if e%10==0:
            loss=0
            #print("weight=%s,bias=%f"%(str(weight),bias))
            for i in range(len(labels)):
                loss += (labels[i] - weight.dot(x[i]) - bias) ** 2
            loss_arr.append(loss/len(labels))
            print("after {%d} epoch ,the loss on train data is {%f}"%(e,loss/len(labels)))

    # keshihua
    plot.title("The loss of the training")
    plot.xlabel('epoch*10')
    plot.ylabel('loss')
    plot.plot(np.array(range(epoch//10)),np.array(loss_arr))
    plot.show()
    return weight,bias
def validate(vol_x,vol_y,weight,bias):
    loss=0
    for i in range(len(vol_y)):
        loss+=(vol_y[i]-(weight.dot(vol_x[i])+bias))**2
    return loss/len(vol_y)

if __name__=='__main__':
    Data=pd.read_csv('./data/train.csv',encoding='gb18030',usecols=range(3,27))
    x,y=DataProcess(Data)
    train_x,train_y=x[0:3200],y[0:3200]
    vol_x,vol_y=x[3200:len(y)],y[3200:len(y)]
    #训练
    weight,bias=train(train_x, train_y, 5000)
    #验证测试
    loss=validate(vol_x,vol_y,weight,bias)
    print("The loss of validation is %f" % loss)

