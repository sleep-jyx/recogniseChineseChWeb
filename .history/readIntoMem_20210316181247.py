import pickle
import logging
import argparse  # 提取命令行参数
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
        #x = x.view(-1, 2704)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    txt_path = './data/picRoad.txt'
    num_class = 3755
    transform = transforms.Compose([transforms.Resize((64, 64)),  # 缩放为正方形；但如果只有一个数，代表最短边缩放到这个数，但形状不变即长宽等比例缩小
                                    transforms.Grayscale(),
                                    transforms.ToTensor()])
    train_set = MyDataset(txt_path, num_class, transform)
    print(train_set.__len__())
    logging.debug('数据加载开始')
    train_loader = DataLoader(
        train_set, batch_size=10, shuffle=True, num_workers=2)  # 批大小=10，双线程加载
    logging.debug('数据加载结束')

    # 查看一个样例
    # (data, label) = train_set[5004]
    # print(type(label))
    # print(label)
    # img = data[0, :, :]
    # print(img.shape)
    # mp.plotbyData(img)

    # 查看一批样例
    '''
    dataiter = iter(train_loader)  # 随机迭代
    images, labels = dataiter.next()  # 返回4张图片及标签
    with open('char_dict', 'rb') as f:
        dict = pickle.load(f)
        # 根据值找键
        print(' '.join('%5s' % list(dict.keys())[
            list(dict.values()).index(labels[j])] for j in range(10)))

        result = transforms.ToPILImage()(tv.utils.make_grid(images)).resize((500, 50))
        result.show()
    '''

    net = NetSmall()  # 神经网络实例化

    '''**************************
        定义损失函数和优化器
    ******************************'''
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降法

    '''**************************
        训练网络
            (1) 输入数据
            (2) 前向传播+反向传播
            (3) 更新参数
    ******************************'''
    torch.set_num_threads(8)  # 多线程
    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            # 输入数据
            inputs, labels = data

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f'
                      % (epoch+1, i+1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
