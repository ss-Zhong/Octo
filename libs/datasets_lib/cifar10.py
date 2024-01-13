import pickle
from . import *
import numpy

LabelsDict = {
    0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
}

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')

    for key in data_dict:
        if key != b'labels' and key != b'data':
            continue

        data = data_dict[key]
        if isinstance(data, bytes):
            data = numpy.frombuffer(data_dict[key], dtype=np.uint8)
        # hint(type(data), key)
        data_dict[key] = np.array(data)

    return data_dict

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
def loadCIFAR10(normalize = True, flatten = False):
    dataset = {}
    dataset['train_img'], dataset['train_label']  = getData(train= True)
    dataset['test_img'], dataset['test_label'] = getData(train=False)

    if normalize:
        for key in ('train_img', 'test_img'):
            cifar_mean = np.array([0.485,0.456,0.406])
            cifar_std = np.array([0.229,0.224,0.225])
            
            # dataset[key] = (dataset[key].astype('float32') / 255.0 - cifar_mean.reshape(1, 3, 1, 1)) / cifar_std.reshape(1, 3, 1, 1)
            dataset[key] = dataset[key].astype('float32') / 255.0
                            
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 3, 32, 32)
    
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])