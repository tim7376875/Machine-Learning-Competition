import numpy as np
import torch.optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
import pandas as pd
import time
from resnet50model import *

'''定義訓練設備'''
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)

'''自定義數據集'''
class mydata(Dataset):
    def __init__(self, root_dir):
        '''設定數據路徑'''
        self.root_dir = root_dir
        self.img_list_dir = os.listdir(self.root_dir)

    def __getitem__(self, idx):
        img_name = self.img_list_dir[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        '''先縮放圖片大小進行切割，再將PIL Image類型轉成Tensor'''
        trans_datasets = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        img_tensor = trans_datasets(img)
        '''回傳Tensor類型的圖像[3, 224, 224]還有label'''
        return img_tensor

    def __len__(self):
        return len(self.img_list_dir)


#"C:/Users/USER/Desktop/pycharm_pytorch/train"
root_dir = "D:/flowertest"
#val_root_dir = "/home/users/EPARC/EPARC2/val_data"
datasets = mydata(root_dir)
#val_datasets = val_data(val_root_dir)

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
print(means)
print(stdevs)


