import os
import myUtils.createFile as cf


def classes_txt(root, out_path, num_class=None):
    '''
    只是把数据文件路径读到输出文件中
    :param root: data set path
    :param out_path: txt file path
    :param num_class: how many classes needed
    :return: None
    '''
    dirs = os.listdir(root)  # 列出数据集目录下所有类别所在文件夹名
    num_class = len(dirs)    # 文件夹数,文字类别数： 训练集0~3754共3755种汉字，接近90万个字
    cf.createFile(out_path)  # 创建输出文件

    with open(out_path, 'r+') as f:
        dirs.sort()
        dirs = dirs[0:num_class]  # 读出所有数据集
        for dir in dirs:  # 列出所有分类目录
            # 列出每个分类文件下(同一个字)下面的所有数据图片文件
            files = os.listdir(os.path.join(root, dir))
            for file in files:
                picRoad = root + '/'+dir+'/'+file
                f.write(picRoad + '\n')  # 把文件目录写入到输出文件里


inPath = '../data/train'
outPath = './data/trainPicRoad.txt'
classes_txt(inPath, outPath)
inPath = '../data/test'
outPath = './data/testPicRoad.txt'
classes_txt(inPath, outPath)
