import os
import logging
logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s %(message)s')

'''
创建文件，路径支持 父文件夹..、 当前文件夹.、 绝对路径
支持直接输入文件名,这样会在当前文件夹下创建该文件
直接只创建文件夹
'''


def createFile(path):
    ''' 
    因为os.mkdir每次只能向下一层创建文件夹，
    要想创建 './a/b/c.txt'要先创建 './a'文件夹，再创建'./a/b'文件夹
    '''
    dirName = path.split('/')  # 提取路径中各级文件夹名，返回列表值，如['.', 'a', 'b','a.txt']
    # 注意,os.mkdir('a')和os.mkdir('./a')和os.mkdir('a/')作用一样，都是在当前文件夹下创建a文件夹
    # 试图创建os.mkdir('.')和os.mkdir('./')效果一样，都是创建当前文件夹，因为已存在，所以会报错
    subDir = ''
    for i in range(len(dirName)-1):  # 假定路径最后一个是有后缀的文件，不作为子文件夹创建
        subDir = subDir + dirName[i] + '/'
        if not os.path.exists(subDir):
            os.mkdir(subDir)

    # 创建文件，但如果路径中不含文件只是为了创建文件夹呢，增加判断最后一个是文件还是文件夹 os.path.isfile(path)
    a = True if '.' in str else False
    filePath = subDir + dirName[len(dirName) - 1]
    if isFile:  # 如果是文件则创建文件
        if not os.path.exists(filePath):  # 输出文件路径不存在就新建
            f = open(filePath, 'w')
            f.close()
        else:
            logging.debug('该路径下文件已存在')
    else:  # 如果是文件夹则创建文件夹
        if not os.path.exists(filePath):
            os.mkdir(filePath)
        else:
            logging.debug('该路径下文件夹已存在')
