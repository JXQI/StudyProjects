from Models.fcn import VGGNet,FCNs
from Loader.segment_loader import dataloader
from Untils.segment_until import Accuracy,iou_mean
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm
from datetime import datetime
from os.path import join


class Process:
    def __init__(self,device,num_worker=0,batch_size=4,lr=0.1,num_class=2,net='ResNet50',\
                 pretrained=False,crop_size=(160,160)):
        self.device = device
        self.batch_size=batch_size
        self.num_worker=num_worker
        self.lr=lr
        self.num_class=num_class
        self.netname=net
        self.crop_size=crop_size
        self.net=FCNs(pretrained_net=VGGNet(pretrained=pretrained),n_class=self.num_class)
        self.net=self.net.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        #self.transform = transforms.Compose([transforms.ToTensor()])
        train_set=dataloader(path='./Data',dataset='./Loader/segment_train',transform=self.transform,crop_size=self.crop_size)
        val_set=dataloader(path='./Data',dataset='./Loader/segment_val',transform=self.transform,crop_size=self.crop_size)
        print("\n训练集数目:%d\t验证集数目:%d\n"%(len(train_set),len(val_set)))
        self.train_loader=DataLoader(dataset=train_set,batch_size=self.batch_size,shuffle=True,num_workers=self.num_worker)
        self.val_loader = DataLoader(dataset=val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
        self.loss=nn.CrossEntropyLoss()
        self.optim=optim.SGD(self.net.parameters(),lr=self.lr,momentum=0.9,weight_decay=0.001)
        #调整学习率
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[200, 200], gamma=0.1)
        #存储结果最好的模型参数
        self.best_model = ''
        # 存储训练结果为.csv文件
        path = './segment_result'
        if not os.path.isdir(path):
            os.mkdir(path)
    def train(self,epoch):
        max_iou = 0
        for j in range(epoch):
            running_loss=0
            prev_time = datetime.now()
            self.net.train()
            tbar=tqdm(self.train_loader)
            correct,all_loss,iou,total=0,0,0,0
            for i,data in enumerate(tbar,0):
                self.optim.zero_grad()
                inputs,labels=data[0].to(self.device),data[1].to(self.device)
                #print(inputs,labels)
                output=self.net(inputs)
                #print(output,labels)
                loss=self.loss(output,labels)
                loss.backward() #计算梯度，反向传播
                self.optim.step()
                #计算准确率
                total+=1
                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum().item()
                all_loss+=loss
                iou += iou_mean(predicted, labels)
                ######
                running_loss+=loss
                if i%10==9:
                    print("[%d, %d] loss:%f"%(j+1,i+1,running_loss/10))
                    running_loss=0
            #验证准确率
            self.net.eval()
            #loss_temp, acc_temp, loss_per,iou_mean = Accuracy(self.net, self.train_loader, self.loss, self.device,crop_size=self.crop_size)
            val_loss_temp, val_acc_temp, val_loss_per, val_iou_mean = Accuracy(self.net, self.val_loader, self.loss, self.device,crop_size=self.crop_size)
            epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f},Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format\
                             (j+1, all_loss/total, correct/total,iou/total ,val_loss_temp, val_acc_temp, val_iou_mean))
            # 计算时间
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
            print('\n'+epoch_str + time_str + ' lr: {}'.format(self.lr)+'\n')

            # 保存所有的model,并且挑出最好的
            model_name = 'Vgg' + '_' + str(j) + '_' + str(int(correct/total * 100)) + '.pth'
            path = './Weights'
            if not os.path.isdir(path):
                os.mkdir(path)
            # torch.save(self.net.state_dict(),join(path,model_name))
            if iou > max_iou:
                max_iou = iou
                self.best_model = 'best_' + model_name
                torch.save(self.net.state_dict(), join(path, self.best_model))
            # 更新学习率：
            self.scheduler.step()
    def validate(self):
        pass
if __name__=="__main__":
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # pro=Process(device)
    # pro.train(epoch=2)
    # pro.validate()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pro = Process(device,batch_size=8,crop_size=(480,480))
    pro.train(epoch=1)