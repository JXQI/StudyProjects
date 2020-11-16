from model import Model
#from loader import dataloader
from loader_mul import dataloader
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from until import Accuracy,drawline,SaveCsv,Draw_ROC,save_predict
from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import operator
from functools import reduce

class Process:
    def __init__(self,device,num_worker,batch_size,lr=0.1,num_class=2,net='Linear_2',\
                 pretrained=False,Weight_path='',isDrop=(False,0.2)):
        self.device = device
        self.batch_size=batch_size
        self.num_worker=num_worker
        self.lr=lr
        self.isDrop=isDrop
        self.num_class=num_class
        self.model=Model(net=net,Weight_path=Weight_path,pretrained=pretrained,isDrop=self.isDrop,num_class=num_class)
        self.net=self.model.Net()
        self.net=self.net.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        train_set=dataloader(path='./data',data_set='train',transforms=self.transform,num_class=num_class)
        val_set=dataloader(path='./data',data_set='val',transforms=self.transform,num_class=num_class)
        print(len(train_set),len(val_set))
        self.train_loader=DataLoader(dataset=train_set,batch_size=self.batch_size,shuffle=True,num_workers=self.num_worker)
        self.val_loader = DataLoader(dataset=val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
        self.loss=nn.CrossEntropyLoss()
        #self.optim=optim.Adam(self.net.parameters(),lr=self.lr)
        self.optim=optim.SGD(self.net.parameters(),lr=self.lr,momentum=0.9,weight_decay=0.001)
        #调整学习率
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[200, 200], gamma=0.1)
        #存储结果最好的模型参数
        self.best_model = ''
        # 存储训练结果为.csv文件
        self.result_csv = SaveCsv(name=['model', 'batch_size', 'lr',"isDrop","epoch","accu","roc_score",'F1_score',"val_accu","val_roc_score",'val_F1_score'], path='./result',file_name=self.net.name+'.csv')
        #生成ROC曲线
        self.roc=Draw_ROC(path='./result',label=self.net.name)
    def train(self,epoch):
        loss_list=[]
        acc_list=[]
        loss_list_val = []
        acc_list_val=[]
        max_acc=0
        self.best_model=''
        running_loss_arr = []
        running_loss_arr_val = []
        for j in range(epoch):
            running_loss=0
            self.net.train()
            for i,data in enumerate(self.train_loader,0):
                self.optim.zero_grad()
                inputs=[0,0]
                inputs[0],inputs[1],labels=data[0][0].to(self.device),data[0][1].to(self.device),data[1].to(self.device)
                #print(inputs,labels)
                output=self.net(inputs)
                #print(output,labels)
                loss=self.loss(output,labels)
                loss.backward() #计算梯度，反向传播
                self.optim.step()
                running_loss+=loss
                if i%10==9:
                    print("[%d, %d] loss:%f"%(j+1,i+1,running_loss/10))
                    #running_loss_arr.append(running_loss/100) #TODO:增加loss曲线的显示
                    running_loss=0
            loss_temp,acc_temp,loss_per,labels,targets,predicts=Accuracy(self.net,self.train_loader,self.loss,self.device)
            val_loss, val_acc, val_loss_arr, val_labels, val_targets, val_predicts = Accuracy(self.net, self.val_loader,self.loss, self.device)
            loss_list.append(loss_temp)
            acc_list.append(acc_temp)
            running_loss_arr.append(loss_per)
            #添加验证集数据
            loss_list_val.append(val_loss)
            acc_list_val.append(val_acc)
            running_loss_arr_val.append(val_loss_arr)

            #计算roc_score
            roc_score=self.roc.roc_score(labels, targets) if self.num_class==2 else 0
            val_roc_score=self.roc.roc_score(val_labels, val_targets) if self.num_class==2 else 0
            #计算F1—score
            F1_score = self.roc.f1_score(labels, predicts) if self.num_class==2 else 0
            val_F1_score=self.roc.f1_score(val_labels, val_predicts) if self.num_class==2 else 0
            print("%d epoch the loss is %f,the train_accuarcy is %f,the val_AUC is %f,the val_F1-score is %f "%(j+1,loss_temp,acc_temp,roc_score,F1_score))
            print("%d epoch the loss is %f,the val_accuarcy is %f,the val_AUC is %f,the val_F1-score is %f " % (j + 1, val_loss, val_acc,val_roc_score,val_F1_score))

            '''存储训练结果 name=['image_size','batch_size','lr',"epoch","accu"]'''
            self.result_csv.savefile(my_list=[self.net.name, self.batch_size,self.lr,self.isDrop, j+1, acc_temp,roc_score,F1_score,val_acc,val_roc_score,val_F1_score])

            #保存所有的model,并且挑出最好的
            model_name='Linear'+'_'+str(j)+'_'+str(int(acc_temp*100))+'.pth'
            path='./Weights'
            if not os.path.isdir(path):
                os.mkdir(path)
            #torch.save(self.net.state_dict(),join(path,model_name))
            if acc_temp>max_acc:
                max_acc=acc_temp
                self.best_model='best_'+model_name
                torch.save(self.net.state_dict(), join(path, self.best_model))
            #更新学习率：
            self.scheduler.step()

        drawline(range(epoch),loss_list,"epoch","loss","the loss of train")
        drawline(range(epoch),acc_list, "epoch","accuarcy", "the accuracy of train")
        running_loss_arr = reduce(operator.add, running_loss_arr)
        drawline(range(len(running_loss_arr)), running_loss_arr, "i", "loss", "the train_loss of the pre data") #TODO:增加loss的显示

        #增加val的loss显示
        drawline(range(epoch), loss_list_val, "epoch", "loss", "the loss of val")
        drawline(range(epoch), acc_list_val, "epoch", "accuarcy", "the accuracy of val")
        running_loss_arr_val = reduce(operator.add, running_loss_arr_val)
        drawline(range(len(running_loss_arr_val)), running_loss_arr_val, "i", "loss",
                 "the val_loss of the pre data")  # TODO:增加loss的显示
        #plt.show()  #TODO:可以改造

    def validate(self):
        self.net.load_state_dict(torch.load(join('./Weights',self.best_model)))
        self.net.eval()
        val_loss,val_acc,val_loss_arr,val_labels,val_targets,val_predicts=Accuracy(self.net,self.val_loader,self.loss,self.device)
        train_loss, train_acc, train_loss_arr,train_labels,train_targets,train_predicts= Accuracy(self.net, self.train_loader, self.loss, self.device)
        #保存预测结果
        save_predict(val_labels, val_predicts, val_targets,filename="val_output.csv")
        save_predict(train_labels,train_predicts,train_targets,filename="train_output.csv")
        # 画出ROC曲线并且保存
        if self.num_class==2:self.roc.ROC(label=val_labels, predict=val_targets,name='val')
        if self.num_class==2:self.roc.ROC(label=train_labels, predict=train_targets,name='train')
        print(len(val_loss_arr),len(train_loss_arr))
        drawline(range(len(val_loss_arr)), val_loss_arr, "i", "loss", "the val_best_loss of the pre data")  # TODO:增加loss的显示
        drawline(range(len(train_loss_arr)), train_loss_arr, "i", "loss", "the train_best_loss of the pre data")  # TODO:增加loss的显示
        val_roc=self.roc.roc_score(val_labels, val_targets) if self.num_class==2 else 0
        val_f1=self.roc.f1_score(val_labels, val_predicts) if self.num_class==2 else 0
        train_roc=self.roc.roc_score(val_labels, val_targets) if self.num_class==2 else 0
        train_f1=self.roc.f1_score(val_labels, val_predicts) if self.num_class==2 else 0
        print("The vol_loss is %f ,The accuarcy is %f,The roc_score is %f,f1_score is %f"%(val_loss,val_acc,val_roc,val_f1))
        print("The train_loss is %f ,The accuarcy is %f,The roc_score is %f,f1_score is %f" % (train_loss, train_acc,train_roc,train_f1))

if __name__=="__main__":
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # pro=Process(device)
    # pro.train(epoch=2)
    # pro.validate()
    path='./Weights'
    if not os.path.isdir(path):
        os.mkdir(path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pro = Process(device,batch_size=8)
    pro.train(epoch=1)