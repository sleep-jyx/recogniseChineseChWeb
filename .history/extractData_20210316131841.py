import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse  # 提取命令行参数
import myUtils.createFile as cf


def classes_txt(root, out_path, num_class=None):
    '''
    write image paths (containing class name) into a txt file.
    :param root: data set path
    :param out_path: txt file path
    :param num_class: how many classes needed
    :return: None
    '''
    dirs = os.listdir(root)  # 列出数据集目录下所有类别所在文件夹名
    num_class = len(dirs)    # 文件夹数： 训练集0~3754共3755个
    cf.createFile(out_path)  # 创建输出文件

    with open(out_path, 'r+') as f:
        dirs.sort()
        dirs = dirs[0:3]  # 读3个试试
        for dir in dirs:  # 列出所有分类
            # 列出每个分类文件下(同一个字)下面的所有数据图片文件
            files = os.listdir(os.path.join(root, dir))
            for file in files:
                f.write(os.path.join(root, dir, file) + '\n')


inPath = '../data/train'
outPath = './data/train/a.txt'
classes_txt(inPath, outPath)