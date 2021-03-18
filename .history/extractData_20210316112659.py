import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse  # 提取命令行参数


def classes_txt(root, out_path, num_class=None):
    '''
    write image paths (containing class name) into a txt file.
    :param root: data set path
    :param out_path: txt file path
    :param num_class: how many classes needed
    :return: None
    '''
    dirs = os.listdir(root)  # 列出根目录下所有类别所在文件夹名
    print(dirs)
    num_class = len(dirs)  # 文件夹数

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    if not os.path.exists(out_path):  # 输出文件路径不存在就新建
        os.mkdir(dir_name)
        f = open(out_path, 'w')
        f.close()
    '''
    if not os.path.exists(out_path):  # 输出文件路径不存在就新建
        f = open(out_path, 'w')
        f.close()
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


'''
inPath = '../data'
outPath = './data/a'
classes_txt(inPath, outPath)
'''
'''
outPath = './data/a'
str = outPath.split('/')
print(str)

os.mkdir('./data/a/b')
#f = open(outPath, 'w')
'''


def createDir(path):
    ''' 
    因为os.mkdir每次只能向下一层创建文件夹，
    要想创建 './a/b/c.txt'要先创建 './a'文件夹，再创建'./a/b'文件夹
    '''
    dirName = path.split('/')  # 提取路径中各级文件夹名，返回列表值，如['.', 'a', 'b','a.txt']
    # 注意,os.mkdir('a')和os.mkdir('./a')作用一样，都是在当前文件夹下创建a文件夹
    subDir = ''
    for i in range(len(dirName)):
        subDir = subDir + "/"+dirName[i]
        # if not os.path.exists(dir_name):
        print(subDir)


path = './a/b/c.txt'
# createDir(path)

os.mkdir('.')
