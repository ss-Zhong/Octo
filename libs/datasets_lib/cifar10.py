from . import *
import pickle
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

        if 'cupy' in str(type(mypy)) and isinstance(data, bytes):
            data = numpy.frombuffer(data_dict[key], dtype=mypy.uint8)

        data_dict[key] = mypy.array(data)

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
                data = mypy.concatenate([data, batch[b'data']])
                labels = mypy.concatenate([labels, batch[b'labels']])

    else:
        batch = unpickle(DatasetsDict + 'cifar-10-py/test_batch')
        data = batch[b'data']
        labels = batch[b'labels']
    return mypy.array(data), mypy.array(labels)

"""
加载cifar10数据集函数
normalize: 是否将图像标准化
flatten: 是否将图像展开
"""
def loadDataset(normalize = True, flatten = False):
    dataset = {}
    dataset['train_img'], dataset['train_label']  = getData(train= True)
    dataset['test_img'], dataset['test_label'] = getData(train=False)

    if normalize:
        for key in ('train_img', 'test_img'):
            cifar_mean = mypy.array([0.485,0.456,0.406])
            cifar_std = mypy.array([0.229,0.224,0.225])

            dataset[key] = (dataset[key].astype('float32') / 255.0 - cifar_mean.reshape(1, 3, 1, 1)) / cifar_std.reshape(1, 3, 1, 1)
            # dataset[key] = dataset[key].astype('float32') / 255.0
                            
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 3, 32, 32)

        horizontal_flip(dataset['train_img'])
    
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])