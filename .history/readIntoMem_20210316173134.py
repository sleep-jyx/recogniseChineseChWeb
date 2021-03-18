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
import logging
import pickle
logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s %(message)s')
logging.disable(logging.DEBUG)


class MyDataset(Dataset):  # 是DataSet的子类，Dataset是pytoch的api，和Dataloader一起使用
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = []  # 存储图片路径
        labels = []  # 存储类别名，在本例中是数字,标签
        # 打开上一步生成的txt文件
        with open(txt_path, 'r') as f:
            for line in f:  # 训练集有89万条记录
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


def justTest():
    txt_path = './data/picRoad.txt'
    num_class = 3755
    transform = transforms.Compose([transforms.Resize((20, 20)),  # 缩放为正方形；但如果只有一个数，代表最短边缩放到这个数，但形状不变即长宽等比例缩小
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    mydataset = MyDataset(txt_path, num_class, transform)
    image, label = mydataset.__getitem__(100)
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
    mp.plotbyData(img)
    # mydataset.showPic(100005)  # 使用matplotlib


class NetSmall(nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()
        # 3个参数分别是in_channels，out_channels，kernel_size，还可以加padding
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 3755)  # 命令行参数，后面解释

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2704)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    txt_path = './data/picRoad.txt'
    num_class = 3755
    transform = transforms.Compose([transforms.Resize((20, 20)),  # 缩放为正方形；但如果只有一个数，代表最短边缩放到这个数，但形状不变即长宽等比例缩小
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    train_set = MyDataset(txt_path, num_class, transform)
    print(train_set.__len__())
    logging.debug('数据加载开始')
    train_loader = DataLoader(
        train_set, batch_size=10, shuffle=True, num_workers=2)  # 批大小=10，双线程加载
    logging.debug('数据加载结束')

    # 查看一个样例
    #(data, label) = train_set[5004]
    # print(type(label))
    # print(label)
    #img = data[0, :, :]
    # print(img.shape)
    # mp.plotbyData(img)

    # 查看一批样例
    dataiter = iter(train_loader)
    images, labels = dataiter.next()  # 返回4张图片及标签
    with open('char_dict', 'rb') as f:
        dict = pickle.load(f)
    # 根据值找键

    # print(' '.join('%11s' % list(data.keys())[
     #     list(data.values()).index(labels[j])] for j in range(10)))
    #show(tv.utils.make_grid((images+1)/2)).resize((400, 100))

    net = NetSmall()  # 神经网络实例化


train()
