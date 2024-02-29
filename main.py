import torch.optim
from torch import nn
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os
import cv2
from PIL import Image
import pandas as pd
import time
#from c_stdmean import *
#from resnet50model import *
from resnetmodel import *
#from vgg16model import *
#from mycNN import *

'''定義訓練設備'''
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)



'''自定義數據集'''
class mydata(Dataset):
    def __init__(self, root_dir):
        '''打開且讀取label.csv文件'''
        self.labels = {}
        with open('label2.csv', newline='') as csvfile:
            data = pd.read_csv(csvfile)
            for j in range(len(data)):
                self.labels[data["filename"][j]] = data["category"][j]

        '''設定訓練集數據路徑'''
        self.root_dir = root_dir
        self.img_list_dir = os.listdir(self.root_dir)

    def __getitem__(self, idx):
        img_name = self.img_list_dir[idx]
        target = self.labels[img_name]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        '''先縮放圖片大小進行切割，再將PIL Image類型轉成Tensor'''
        trans_datasets = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(size=((224, 224)), scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
            #transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(degrees=30),
            #transforms.FiveCrop(224),
            #transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.481, 0.423, 0.368], std=[0.247, 0.241, 0.249])(crop) for crop in crops])),
            #transforms.CenterCrop(224),
            transforms.ColorJitter(0.2, 0.2, 0.4),
            #transforms.GaussianBlur(9, 5.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.423, 0.368], std=[0.247, 0.241, 0.249]),
        ])
        img_tensor = trans_datasets(img)
        '''回傳Tensor類型的圖像[3, 224, 224]還有label'''
        return img_tensor, target

    def __len__(self):
        return len(self.img_list_dir)

'''自定義驗證數據集 8:2'''
class val_data(Dataset):
    def __init__(self, val_root_dir):
        '''打開且讀取val_set.csv文件'''
        self.val_labels = {}
        with open('val_set.csv', newline='') as csvfile:
            val = pd.read_csv(csvfile)
            for k in range(len(val)):
                self.val_labels[val["filename"][k]] = val["category"][k]

        '''設定驗證集數據路徑'''
        self.val_root_dir = val_root_dir
        self.val_list_dir = os.listdir(self.val_root_dir)

    def __getitem__(self, idx):
        val_name = self.val_list_dir[idx]
        val_target = self.val_labels[val_name]
        val_item_path = os.path.join(self.val_root_dir, val_name)
        val = Image.open(val_item_path)
        '''先縮放圖片大小進行切割，再將PIL Image類型轉成Tensor'''
        trans_datasets = transforms.Compose([
            transforms.Resize(224),
            #transforms.RandomCrop(224),
            #transforms.RandomHorizontalFlip(0.5),
            transforms.FiveCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.476, 0.422, 0.364], std=[0.246, 0.241, 0.248])(crop) for crop in crops])),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            #transforms.Normalize(mean=[0.476, 0.422, 0.364], std=[0.246, 0.241, 0.248]),
        ])
        val_tensor = trans_datasets(val)
        '''回傳Tensor類型的圖像[3, 224, 224]還有label'''
        return val_tensor, val_target

    def __len__(self):
        return len(self.val_list_dir)


#"C:/Users/USER/Desktop/pycharm_pytorch/train"
root_dir = "/home/users/EPARC/EPARC2/train"
val_root_dir = "/home/users/EPARC/EPARC2/val_data"
datasets = mydata(root_dir)
val_datasets = val_data(val_root_dir)

'''利用dataloader來做數據集批量訓練及驗證'''
dataloader = DataLoader(dataset = datasets, batch_size = 10, num_workers = 4, drop_last = False)
val_dataloader = DataLoader(dataset = val_datasets, batch_size = 2, num_workers = 4, drop_last = False)

'''搭建resnet18神經網路'''
#res50 = ResNet50()
res18 = ResNet18()
res18.to(device)
#res50.to(device)
#vgg16 = myVGG16()
#vgg16.to(device)
#my = MyCNN()
#my.to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

parameter_count = count_parameters(res18)
print(f"#parameters:{parameter_count}")

'''損失函數'''
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

'''創建優化器'''
learning_rate = 0.01126
#optimizer = torch.optim.Adam(res18.parameters(), lr=learning_rate, weight_decay=0.001)
'''進行L2正則化防止過擬合, weight_decay=0.001'''
optimizer = torch.optim.SGD(res18.parameters(), lr = learning_rate, momentum=0.9, weight_decay=0.005)
#scheduler = StepLR(optimizer, step_size = 10, gamma = 0.1, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode='min', verbose=True, min_lr=1e-5)

'''設定訓練網路之參數'''
'''紀錄訓練次數'''
total_train_step = 0
'''訓練輪數'''
epoch = 500
'''計算累積花費時間'''
start_time = time.time()

for i in range(epoch):
    print("-------------第{}輪訓練開始--------------".format(i + 1))
    '''每輪參數初始化'''
    train_loss = 0.0
    total_train_acc = 0.0
    val_loss = 0.0
    total_val_acc = 0.0

    '''訓練開始'''
    res18.train()
    for data in dataloader:
        imgs, targets = data
        #bs, nc, c, h, w = imgs.size()
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = res18(imgs)
        #outputs = res18(imgs.view(-1, c, h, w))
        #outputs_avg = outputs.view(bs, nc,-1).mean(1)
        loss = loss_fn(outputs, targets)
        #loss = loss_fn(outputs_avg, targets)
        '''優化器模型'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        total_train_step = total_train_step + 1
        '''計算模型損失'''
        train_loss = train_loss + loss
        '''計算訓練準確率'''
        #_, train_acc = torch.max(outputs_avg.data, 1)
        train_acc = (outputs.argmax(1) == targets).sum()
        total_train_acc = total_train_acc + train_acc
        '''每訓練100次則印出一次當前結果'''
        if (total_train_step % 100 == 0):
            print("訓練次數:{}, Loss:{}".format(total_train_step, loss.item()))

    '''資料驗證集'''
    if (len(val_datasets) > 0):
        res18.eval()
        '''停止更新梯度'''
        with torch.no_grad():
            for data in val_dataloader:
                vals, val_targets = data
                bs, nc, c, h, w = vals.size()
                vals, val_targets = vals.to(device), val_targets.to(device)
                val_outputs = res18(vals.view(-1, c, h, w))
                #val_outputs = res18(vals)
                val_outputs_avg = val_outputs.view(bs, nc, -1).mean(1)
                #vloss = loss_fn(val_outputs, val_targets)
                vloss = loss_fn(val_outputs_avg, val_targets)
                val_loss = val_loss + vloss
                #val_acc = (val_outputs.argmax(1) == val_targets).sum()
                val_acc = (val_outputs_avg.argmax(1) == val_targets).sum()
                total_val_acc = total_val_acc + val_acc

    scheduler.step(val_loss)
    #scheduler.step()

    end_time = time.time()
    print("經過{}輪所累積花費時間:{}秒".format((i + 1), (end_time - start_time)))
    print("本輪訓練準確率:", total_train_acc.item() / len(datasets))
    print("本輪驗證準確率:", total_val_acc.item() / len(val_datasets))
    print("本輪驗證總損失:", val_loss.item())
    print("本輪訓練總損失:", train_loss.item())

    '''儲存本輪所訓練之結果'''
    if(i == 79):
        #torch.save(res18, "res18_{}.pth".format(i))
        print("模型已保存")


'''
算自定義數據集的標準化
means = [0, 0, 0]
stdevs = [0, 0, 0]
for data in datasets:
    img = data[0]
    for i in range(3):
        # 一个通道的均值和标准差
        means[i] += img[i, :, :].mean()
        stdevs[i] += img[i, :, :].std()

means = np.asarray(means) / len(datasets)
stdevs = np.asarray(stdevs) / len(datasets)

imgs, target = datasets[0]
for i in dataloader:
    imgs, targets = i
    print(imgs.shape)
    print(targets)
'''

import math

import torch
import torch.nn as nn
import torchvision

def conv3x3(in_channels,out_channels,stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self,in_channels,out_channels,stride=1,downsample=None):
		super(BasicBlock,self).__init__()
		self.conv1 = conv3x3(in_channels,out_channels,stride) #大小不變
		self.bn1 = nn.BatchNorm2d(out_channels) # 批標準化層
		self.relu = nn.LeakyReLU(inplace=True) # 激活函數

		self.conv2 = conv3x3(out_channels,out_channels) #大小不變
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample = downsample # 這個是shortcut
		self.stride = stride

	def forward(self,x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=219):
        super(ResNet, self).__init__()
        self.in_channels = 64  #每一個殘差塊塊輸入深度

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #224-5+4/2+1=112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 縮小1/2
        #112-3+2/2+1=56
        self.layer1 = self._make_layer(block, 64, layers[0])  # 特徵圖大小不變
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 56-1/2+1=28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 28-1/2+1=14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 14-1/2+1=7
        #self.layer5 = self._make_layer(block, 512, layers[4], stride=2)  # 7-1/2+1=4


        #self.avgpool = nn.AvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # 平均池化
        #self.fc = nn.Linear(512 * 2 * 2, 1024),
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

        # 網絡的參數初始化
        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None  # shortcut
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(#64,64,1,2,
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion  # 改變下面的殘差塊的輸入深度

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x) #224-7+6/2+1=112
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)  #112-3+2/2+1=56
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # 將原有的多維輸出拉回一維
        #out = self.fc(out)
        out = self.classifier(out)

        return out

def ResNet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def BResNet18(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2], **kwargs)
    return model
'''
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=219):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False) #224-7+6/1+1=224
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #224-3+2/2+1=113
        self.layer1 = self._make_layer(block, 64, layers[0]) #113-1/1+1=113
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 512 * 10),
            nn.BatchNorm1d(512 * 10),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512 * 10, 512 * 10),
            nn.BatchNorm1d(512 * 10),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512 * 10, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # 卷積參數變量初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) # BN參數初始化
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: #1
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gelu(out)

        return out
'''
if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = ResNet18()
    #print(model)

    input = torch.randn(64, 3, 224, 224)
    out = model(input)
    print(out.shape)