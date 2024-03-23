from .. import *

DatasetsDict = 'F:/Datasets/'

def horizontal_flip(images):
    batch_size, _, _, _ = images.shape
    flip_mask = mypy.random.random(size=batch_size) < 0.5
    images[flip_mask] = images[flip_mask, :, :, ::-1]

    return images

hint("======= Lib of Datasets is Loaded =======")