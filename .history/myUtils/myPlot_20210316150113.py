import matplotlib.pyplot as plt


def plot(path):
    im = plt.imread(path)
    plt.imshow(im)
    plt.show()
