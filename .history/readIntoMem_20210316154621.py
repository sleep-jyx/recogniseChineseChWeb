import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import myUtils.myPlot as mp
import numpy as np
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

    def showPic(self, index):
        mp.plot(self.images[index])

    def __len__(self):
        return len(self.labels)


txt_path = './data/picRoad.txt'
num_class = 3755
transform = transforms.Compose([transforms.Resize((20, 20)),  # 缩放为正方形；但如果只有一个数，代表最短边缩放到这个数，但形状不变即长宽等比例缩小
                                transforms.Grayscale(),
                                transforms.ToTensor()])

mydataset = MyDataset(txt_path, num_class, transform)
image, label = mydataset.__getitem__(100005)
'''
img = image[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
img = img.numpy()  # FloatTensor转为ndarray
img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
# 显示图片
plt.imshow(img)
plt.show()
# image.show()  # 使用Pillow
'''
print(image.size())
img = image[0, :, :]
print(img.shape)
# mydataset.showPic(100005)  # 使用matplotlib
