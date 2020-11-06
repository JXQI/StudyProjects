#encoding=utf-8
import torch
from send_mail import sentemail
import argparse
from train import Process
import matplotlib.pyplot as plt

if __name__=='__main__':
    #Function(sys.argv)
    parse=argparse.ArgumentParser(description="train or test")
    parse.add_argument('--net', type=str,default='ConvNet',help='select the model to train and test')
    parse.add_argument('--pretrained', type=bool, default=False, help='if model pretrained')
    parse.add_argument('--Weight_path', type=str, default="./Weights/*", help='add the pre_Weight')
    parse.add_argument('--isDrop', type=bool,default=True, help='if add the dropout layer and the probility')
    parse.add_argument('--train', type=str,default='train', help='train the model')
    parse.add_argument('--epoch', type=int, default=10, help='the epoch')
    parse.add_argument('--batch_size', type=int, default=4, help='the epoch')
    parse.add_argument('--num_worker', type=int, default=0, help='the num_workers')
    parse.add_argument('--lr', type=float, default=0.000001, help='the learning rate')
    parse.add_argument('--class_type', type=str, default="B", help='task "B" or "T"')
    args=parse.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    pro = Process(device,batch_size=args.batch_size,lr=args.lr,class_type=args.class_type,\
                  net=args.net,pretrained=args.pretrained,Weight_path=args.Weight_path,isDrop=args.isDrop)
    pro.train(epoch=args.epoch)
    pro.validate()
    #plt.show()  # TODO:可以改造
    sentemail()