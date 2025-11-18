#encoding=utf-8
import torch
from Untils.send_email import sentemail
import argparse
from Train.train import Process
import numpy as np
import random

#设置随机种子
SED=7
torch.manual_seed(SED) # cpu
torch.cuda.manual_seed(SED) #gpu
np.random.seed(SED) #numpy
random.seed(SED) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

if __name__=='__main__':
    #Function(sys.argv)
    parse=argparse.ArgumentParser(description="train or test")
    parse.add_argument('--net', type=str,default='ResNet50',help='select the model to train and test')
    parse.add_argument('--pretrained', type=bool, default=False, help='if model pretrained')
    parse.add_argument('--epoch', type=int, default=5, help='the epoch')
    parse.add_argument('--batch_size', type=int, default=4, help='the epoch')
    parse.add_argument('--num_worker', type=int, default=0, help='the num_workers')
    parse.add_argument('--lr', type=float, default=0.000001, help='the learning rate')
    parse.add_argument('--num_class', type=int, default=2, help='class task 2 or 3')
    args=parse.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    pro = Process(device,batch_size=args.batch_size,num_worker=args.num_worker,lr=args.lr,num_class=args.num_class,
                  net=args.net,pretrained=args.pretrained)
    pro.train(epoch=args.epoch)
    pro.validate()
    #plt.show()  # TODO:可以改造
    sentemail()