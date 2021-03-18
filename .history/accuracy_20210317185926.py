import pickle
import logging
import numpy as np
import myUtils.myPlot as mp
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage  # 可以把Tensor转成Image，方便可视化
logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s %(message)s',
                    filename='running.log',
                    filemode='a')
logging.disable(logging.DEBUG)

'''
卷积神经网络类：
    输入层->卷积层1->激活层->池化层
        ->卷积层2->激活层->池化层
        ->全连接层1->全连接层2->全连接层3->输出
    
    输入层：(1,64,64)   1指的是单通道(非RGB彩色图),两个64分别是输入数据的高和宽，即输入图片是64*64大小的汉字图片
    卷积层1:卷积核(1,6,3)    1是输入层通道数，6是卷积核个数或下一层输入的通道数，3是卷积核的边长
    经过卷积层1，输入层变为(6,62,62),再使用relu激活
    池化层：大小(2*2)，把(6,62,62)池化为(6,31,31)
    卷积层2:卷积核(6,16,5),经过卷积层2，数据大小从(6,31,31)变为(16,27,27)，再使用relu激活
    池化层：(16,27,27)池化为(16,13,13)
    全连接层1：将(16,13,13)展开为一维向量(1*2704)，(1*2704)*(2704*512) = (1*512)
    全连接层2：（1*512）*（512*84）=(1*84)
    全连接层3：（1*84）*（84,3755）=(1*3755)
    输出层:(1*3755)，正好对应3755个汉字，哪个值最大，就是预测的值
'''


class NetSmall(nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()
        # 3个参数分别是in_channels，out_channels，kernel_size，还可以加padding
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(2704, 512)
        self.fc2 = nn.Linear(512, 84)
        self.fc3 = nn.Linear(84, 3755)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 2704)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyDataset(Dataset):  # 数据集类，继承pytorch的Dataset，便于后面使用Dataloader
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


if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((64, 64)),  # 缩放为正方形；但如果只有一个数，代表最短边缩放到这个数，但形状不变即长宽等比例缩小
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    txt_path = './data/picRoad.txt'
    num_class = 3755
    train_set = MyDataset(txt_path, num_class, transform)
    train_loader = DataLoader(
        train_set, batch_size=10, shuffle=True, num_workers=2)  # 批大小=10，双线程加载

    model = NetSmall()
    # 加载检查点
    ch_path = 'checkpoint_3_epoch.pkl'
    checkpoint = torch.load(ch_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            #correct += sum(int(predict == labels)).item()
            correct += (predict == labels).sum().item()

            if i % 100 == 99:
                print('batch: %5d,\t acc: %f' % (i + 1, correct / total))
    print('Accuracy: %.2f%%' % (correct / total * 100))
