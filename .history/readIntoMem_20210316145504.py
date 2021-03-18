import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse  # 提取命令行参数


class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = []  # 存储图片路径
        labels = []  # 存储类别名，在本例中是数字,标签
        # 打开上一步生成的txt文件
        with open(txt_path, 'r') as f:
            for line in f:
                # 倒数第二个分隔符是类别，只读取前 num_class 个类
                if int(line.split('/')[-2]) >= num_class:
                    break
                line = line.strip('\n')     # 剪除换行符
                images.append(line)         # 图片路径写入内存
                labels.append(int(line.split('/')[-2]))  # 标签写入内存
        self.images = images
        self.labels = labels
        self.transforms = transforms  # 图片需要进行的变换，ToTensor()等等

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')  # 用PIL.Image读取图像
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)  # 进行变换
        return image, label

    def __len__(self):
        return len(self.labels)


txt_path = './data/picRoad.txt'
num_class = 5
mydataset = MyDataset(txt_path, num_class)
image, label = mydataset.__getitem__(2)
print(image)
image.show()  # 这里使用的是Pillow包即PIL，还可以用opencv
print(label)
