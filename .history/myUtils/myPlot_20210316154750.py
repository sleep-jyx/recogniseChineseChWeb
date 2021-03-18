import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot(path):
    im = plt.imread(path)
    plt.imshow(im)
    plt.show()


def plotbyData(img):
    plt.imshow(im)
    plt.show()
