from . import *

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - mypy.max(x, axis=0)
        y = mypy.exp(x) / mypy.sum(mypy.exp(x), axis=0)
        return y.T 

    x = x - mypy.max(x)
    return mypy.exp(x) / mypy.sum(mypy.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -mypy.sum(mypy.log(y[mypy.arange(batch_size), t] + 1e-7)) / batch_size