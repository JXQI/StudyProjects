from model import Model
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scipy.io import loadmat
import os
from os.path import  join
from collections import defaultdict
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from pandas import DataFrame
import torch.nn.functional as F
from data_deal import deal_Na,read_feature,normalize,write_feature
from tqdm import tqdm
from settings import AGE_MEAN
from torch.utils.data import Dataset


#加载数据
class data_load:
    def __init__(self,filename,save_path):
        self.file=loadmat(filename, mat_dtype=True)
        # 生成之前先删除之前文件中存在的
        self.path = save_path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        self.dictory = read_feature("./dictory.csv")
    # 脑内主要纤维束的特征
    # 提取每一个指标：这里将每个病人的特征单独存储一个文件，也就是说每个总共有700个文件，每个8*100
    def AFQ_feature(self):
        feature_keys = ['FA', 'MD', 'RD', 'AXD', 'liner', 'Curvature', 'torsion', 'volume']
        d_feature = defaultdict(list)
        get_feature=[]
        tbar=tqdm(range(125))
        for index in tbar:
            # 每个病人的特征分开保存
            for i in range(len(feature_keys)):
                # 八个特征分开
                for j in range(20):
                    # 20个纤维束
                    # print(np.around(d['train_set'][j][0][i][index],decimals=3))
                    d_feature[feature_keys[i]].append(np.around(self.file['test_set'][j][0][i][index], decimals=3))
            #处理NAN值和进行标准化
            feature=np.array(list(d_feature.values()))
            feature=deal_Na(features=feature,dictory=self.dictory)
            feature=normalize(feature)
            get_feature.append(feature)
            #write_feature(feature=feature, name=join('test', "0_" + str(index) + '.csv'))
            # 需要注意这里添加进文件之后一定得将字典清空，不然会一直append到每个keys，导致最后一个文件累加进去了前边所有的值
            d_feature.clear()
        return get_feature

    # 对年龄性别等指标归类，保存单个数据的特征，比如年龄性别等等
    def single_dimenon_feature(self):
        d_result = {}
        d_result_keys = ['test_population', 'test_sites']
        for i in self.file.keys():
            if i in d_result_keys:
                if i == 'test_population':
                    temp1,temp2=[],[]
                    for j in range(len(self.file[i][:,0])):
                        temp1.append([self.file[i][:, 0][j]])
                        temp2.append([self.file[i][:, 1][j]])
                    d_result['sex'] = temp1
                    d_result['age'] = temp2
                else:
                    d_result[i] = self.file[i].flatten()
        #pd.DataFrame.from_dict(data=d_result, orient='index').to_csv(join(self.path, 'data_result.csv'))
        return list(zip(d_result['sex'], d_result['age']))

def load_test_data(self_transforms=transforms.Compose([transforms.ToTensor()]),filename='MCAD_AFQ_test.mat', save_path='./test',Is_18=False):
    test_data = data_load(filename, save_path)
    print("数据加载，并且进行预处理！")
    get_features=test_data.AFQ_feature()
    age_sex_data=test_data.single_dimenon_feature()
    print("数据加载完成，预处理完成！")
    data = []
    for item in range(len(get_features)):
        features = get_features[item]     #包含每个病人的纤维素特征
        all_feature = []  # 包含每个病人的纤维素特征和年龄性别特征
        age_sex_feature=age_sex_data[item]
        #这里求（年龄值-平均值）/平均值 的结果作为特征输入
        age_sex_feature[1][0]=(age_sex_feature[1][0]-AGE_MEAN)/AGE_MEAN

        features = np.around(np.array(features, dtype=np.float32), decimals=3)
        if Is_18: features = features[:6] + features[8:]
        age_sex_feature=np.array(age_sex_feature,dtype=np.float32)
        features = features.transpose((1, 2, 0))  # TODO:需用补充一下转换的时候各种转置关系，以及和torch的转换关系
        if self_transforms:
            features = self_transforms(features)
            age_sex_feature=torch.tensor(age_sex_feature)
        all_feature.append(features)
        all_feature.append(age_sex_feature)
        data.append(all_feature)
    return data

class dataloader(Dataset):
    def __init__(self,transforms=None,num_class=3):
        self.transforms=transforms
        self.num_class=num_class
        #self.class_d={"NC":1,"MCI":2,'AD':3}        #TODO:这里需要支持三分类
        self.features=[]
        self.data=load_test_data(self_transforms=self.transforms)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        features=self.data[item]
        return features

class test():
    def __init__(self, device, num_worker, batch_size, num_class=3, net='Linear_2', \
                 pretrained=False, Weight_path='', isDrop=(False, 0.2)):
        self.device = device
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.isDrop = isDrop
        self.num_class = num_class
        self.model = Model(net=net, Weight_path=Weight_path, pretrained=pretrained, isDrop=self.isDrop,
                           num_class=num_class)
        self.net = self.model.Net()
        self.net = self.net.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])
        test_set = dataloader(transforms=self.transform, num_class=num_class)

        print("\n测试集数量:%f\n"%len(test_set))
        self.test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_worker)
    def result(self):
        self.net.eval()
        result={"number":[],"predict":[],"AD_probility":[],"MCI_probility":[],"NC_probility":[]}

        print("\n写入测试结果中...\n")
        tbar_loader=tqdm(self.test_loader)
        with torch.no_grad():
            for i, data in enumerate(tbar_loader, 0):
                inputs = [0, 0]
                inputs[0], inputs[1] = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs, 1)
                # 将label和概率添加进列表中去
                # self.class_d={"NC":1,"MCI":2,'AD':3}
                target={"0":1,"1":2,'2':3}
                for lp in range(len(outputs)):
                    result["number"].append(i*len(outputs)+lp+1)
                    result["AD_probility"].append(round(float(F.softmax(outputs[lp], dim=0)[2]),3)) #TODO:修改的符合要求
                    result["MCI_probility"].append(round(float(F.softmax(outputs[lp], dim=0)[0]),3))  # TODO:修改的符合要求
                    result["NC_probility"].append(round(float(F.softmax(outputs[lp], dim=0)[1]),3))  # TODO:修改的符合要求
                    result["predict"].append(target[str(int(predicted[lp]))])
                    #result["predict"].append((int(predicted[lp])))
        #保存结果
        data = [result["number"], result["predict"], result["AD_probility"],result["MCI_probility"],result["NC_probility"]]
        data = np.array(data).transpose()
        DataFrame(data=data, columns=["被测序号", "预测标签", "AD类概率","NC概率","MCI概率"]).to_csv("模型评估_三人行_模型1.csv", index=False)

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    T=test(device,num_worker=0,batch_size=32,num_class=3,net='ConvNet',Weight_path='./Weights/best_Linear_0_34.pth')
    T.result()