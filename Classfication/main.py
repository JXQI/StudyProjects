import numpy as np
import os
from matplotlib import pyplot as plt

def Data_Deal(train,label,test):
    #加载数据集
    # with open(train) as f:
    #     for i,data in enumerate(f.readlines()):
    #         if i!=0:
    #             train_x.append(data.strip('\n').split(','))
    #     train_x=np.array(train_x,dtype=float)
    #     print(train_x)
    #     print(train_x.shape)
    #简单的写法
    with open(train) as f:
        next(f)
        train_x=np.array([line.strip('\n').split(',')[1:] for line in f],dtype=float)
    print(train_x.shape)
    with open(label) as f:
        next(f)
        label_x=np.array([line.strip('\n').split(',')[1:] for line in f],dtype=float)
    print(label_x.shape)
    with open(test) as f:
        next(f)
        test=np.array([line.strip('\n').split(',')[1:] for line in f],dtype=float)
    print(test.shape)

    return train_x,label_x,test
#激活函数
def _sigmoid(z):
    return np.clip(1/(1.0+np.exp(-z)),1e-8,1-(1e-8))

def model(x,w,b):
    #定义一个线性函数，并且通过一个激活函数,输出在0-1的值
    return _sigmoid(np.matmul(x,w)+b)
#十字交叉熵函数
def _cross_entropy_loss(y_pred,y_lable):
    cross_entropy=-(np.dot(np.transpose(y_lable),np.log(y_pred))+\
                    np.dot(np.transpose(1-y_lable),np.log(1-y_pred)))
    return cross_entropy
#计算梯度
def _gradient(x,label,w,b):
    #w_g=-np.sum((label-model(x,w,b)).dot(x))  #特别注意这里不是卷积
    w_g=-np.sum((label-model(x,w,b).reshape(label.shape))*x)  #TODO：这里为什么需要求X的转置
    b_g=-np.sum(label-model(x,w,b))
    return w_g,b_g
#计算优化器，学习率
#global w_g_sum
#global b_g_sum
#w_g_sum=0
#b_g_sum=0
#def _Adagrad(w_g,b_g):

#打乱数据顺序
def _shuffle(x,y):
    shuffle_temp=np.arange(len(x)) #生成随机种子
    np.random.shuffle(shuffle_temp)
    return x[shuffle_temp],y[shuffle_temp]

def train_dev_spilt(x,label,ratio):
    # 返回train_x,train_label,val_x,val_label
    train_size=int(len(x)*ratio)    #注意此处len()对于nump也是可以用的
    return x[:train_size],label[:train_size],x[train_size:],label[train_size:]
#求准确率
def accuary(pre,label):
    #return 1-np.mean(np.abs(pre-label))
    #print(np.round(pre),label.reshape(pre.shape))
    pre=np.round(pre)
    label=label.reshape(pre.shape)  #注意此处的输出值
    return sum([1 if pre[i]==label[i] else 0 for i in range(len(pre))])/len(pre)

if __name__=='__main__':
    #数据加载
    x,label,test=Data_Deal('./data/X_train','./data/Y_train','./data/X_test')
    #将数据划分为训练集和验证集
    #train_x,train_label,val_x,val_label=train_dev_spilt(x,label,0.1)
    train_x,train_label,val_x,val_label=train_dev_spilt(x,label,0.2)
    #TODO:注意这里内存不够，只取出来验证集的一部分
    val_x, val_label,_,_= train_dev_spilt(val_x, val_label, 0.2)
    print(train_x.shape,train_label.shape,val_x.shape,val_label.shape)
    #打乱训练数据
    train_x,train_label=_shuffle(train_x,train_label)
    #初始化参数
    w=np.zeros(train_x.shape[1])
    b=np.zeros(1)

    #求梯度的和
    w_g_sum=np.zeros(w.shape)
    b_g_sum=np.zeros(b.shape)

    epoch=1000
    lr=0.1
    batch_size=8
    train_accu=[]
    train_loss=[]
    val_acc=[]
    val_loss=[]
    w_store=[]
    b_store=[]
    for j in range(epoch):
        for i in range(int(np.floor(train_x.shape[0]/batch_size))):
            x=train_x[i*batch_size:(i+1)*batch_size]
            y=train_label[i*batch_size:(i+1)*batch_size]
            w_g=0
            b_g=0
            w_g,b_g=_gradient(x,y,w,b)
            w_g_sum+=w_g**2
            b_g_sum+=b_g**2
            w-=lr/w_g_sum**0.5*w_g
            b-=lr/b_g_sum**0.5*b_g
        y_pre=model(train_x,w,b)
            #y_pre=np.round(y_pre)   #TODO: 将数据转换成bool类型
        loss=np.sum(_cross_entropy_loss(y_pre,train_label))/len(train_x)
        acc=accuary(y_pre,train_label)
        train_loss.append(loss)
        train_accu.append(acc)
        w_store.append(w)
        b_store.append(b)

        #处理验证集
        val_y_pre=model(val_x,w,b)
        loss_v = np.sum(_cross_entropy_loss(val_y_pre, val_label)) / len(val_x)
        acc_v = accuary(val_y_pre, val_label)
        val_loss.append(loss_v)
        val_acc.append(acc_v)
        print("the %d epoch loss is on the train_set is %f ,on the val_set is %f" % (j,loss, loss_v))
    print("finished train! the train mean of the loss is %f , the accuracy is %f "%(float(np.mean(loss)),np.max(train_accu)))
    print("finished train! the val mean of the loss is %f , the accuracy is %f " % (float(np.mean(loss_v)), np.max(val_acc)))

    plt.title("the loss of train/val with epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(epoch),train_loss,color='red',label='train')
    plt.plot(range(epoch), val_loss,color='green',label='val')
    plt.legend()
    plt.show()
    plt.title("the accuray of train/val with epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(range(epoch), train_accu,color='red',label='train')
    plt.plot(range(epoch), val_acc,color='green',label='val')
    plt.legend()
    plt.show()





