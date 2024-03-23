from enum import Enum, unique
from libs.datasets_lib import cifar10, fashion_mnist, mnist

@unique
class dataset_(Enum):
    cifar10 = 0
    fashion_mnist = 1
    mnist = 2

# 根据数据库类型依次返回数据、维度、类型数
def loadDataset(dataset, normalize = True, flatten = False):
    if dataset == dataset_.cifar10:
        return cifar10.loadDataset(normalize, flatten), [3, 32], 10
    
    elif dataset == dataset_.fashion_mnist:
        return fashion_mnist.loadDataset(normalize, flatten), [1, 28], 10
    
    elif dataset == dataset_.mnist:
        return mnist.loadDataset(normalize, flatten), [1, 28], 10