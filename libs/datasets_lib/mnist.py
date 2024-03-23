from . import *
import gzip

def getData(train=False):
    path = DatasetsDict + 'mnist/'
    if train:
        labels_path = path + 'train-labels-idx1-ubyte.gz'
        images_path = path + 'train-images-idx3-ubyte.gz'
    else:
        labels_path = path + 't10k-labels-idx1-ubyte.gz'
        images_path = path + 't10k-images-idx3-ubyte.gz'

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = mypy.frombuffer(lbpath.read(), dtype=mypy.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = mypy.frombuffer(imgpath.read(), dtype=mypy.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

"""
加载mnist数据集函数
normalize: 是否将图像标准化
flatten: 是否将图像展开
"""
def loadDataset(normalize = True, flatten = False):
    dataset = {}
    dataset['train_img'], dataset['train_label']  = getData(train= True)
    dataset['test_img'], dataset['test_label'] = getData(train=False)

    if normalize:
        for key in ('train_img', 'test_img'):
            mean = 0.5
            std = 0.5

            dataset[key] = (dataset[key].astype('float32') / 255.0 - mean) / std
            # dataset[key] = dataset[key].astype('float32') / 255.0
                            
    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)
    
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])