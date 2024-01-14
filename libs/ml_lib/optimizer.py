from . import *

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        params -= self.lr * grads

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:                         
            self.v = mypy.zeros_like(grads)
                
        self.v = self.momentum * self.v + self.lr * grads 
        params -= self.v