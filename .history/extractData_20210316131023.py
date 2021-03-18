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
    dirs = os.listdir(root)  # 列出根目录下所有类别所在文件夹名
    # print(dirs)
    num_class = len(dirs)  # 文件夹数
    print('当前文件下文件数', num_class)
    cf.createFile(out_path)  # 创建文件
    with open(out_path, 'r+') as f:
        print(f.readlines())
    # print(end)
    '''
	# 如果文件中本来就有一部分内容，只需要补充剩余部分
	# 如果文件中数据的类别数比需要的多就跳过
    with open(out_path, 'r+') as f:
        try:
            end = int(f.readlines()[-1].split('/')[-2]) + 1
        except:
            end = 0
        if end < num_class - 1:
            dirs.sort()
            dirs = dirs[end:num_class]
            for dir in dirs:
                files = os.listdir(os.path.join(root, dir))
                for file in files:
                    f.write(os.path.join(root, dir, file) + '\n')
'''


inPath = '../data/train'
outPath = './data/train/abt'
classes_txt(inPath, outPath)
