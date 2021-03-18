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


if __name__ == '__main__':

    model = NetSmall()
    # 加载参数
    ch_path = 'checkpoint_3_epoch.pkl'
    checkpoint = torch.load(ch_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    inputs = 
    outputs = model(inputs)
    _, predict = torch.max(outputs.data, 1)
    total += labels.size(0)
    #correct += sum(int(predict == labels)).item()
    correct += (predict == labels).sum().item()
