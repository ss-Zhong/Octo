import pickle
import numpy as np


DatasetsDict = 'F:/Datasets/'
LabelsDict = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def getData(train=False):
    data = None
    labels = None
    if train == True:
        for i in range(1, 6):
            batch = unpickle(DatasetsDict + 'cifar-10-py/data_batch_' + str(i))
            if i == 1:
                data = batch[b'data']
                labels = batch[b'labels']
            else:
                data = np.concatenate([data, batch[b'data']])
                labels = np.concatenate([labels, batch[b'labels']])

    else:
        batch = unpickle(DatasetsDict + 'cifar-10-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    return np.array(data), np.array(labels)


"""
加载cifar10数据集函数
normalize: 是否将图像标准化(标准化到0~1之间)
flatten: 是否将图像展开
"""
def loadCIFAR10(normalize = False, flatten = False):
    dataset = {}
    dataset['train_img'], dataset['train_label']  = getData(train= True)
    dataset['test_img'], dataset['test_label'] = getData(train=False)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] /= 255
    
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 3, 32, 32)
    
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])