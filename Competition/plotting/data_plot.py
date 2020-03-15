from matplotlib import pyplot as plt


def scatter_basic(self, data, feature, label):
    plt.xlabel(feature)
    plt.ylabel(label)
    plt.scatter(data[feature], data[label])
    x0 = 0
    y0 = 0
    x1 = max(data[feature])
    y1 = max(data[label])
    plt.plot([x0, x1], [y0, y1], c='r')
    plt.show()
    return