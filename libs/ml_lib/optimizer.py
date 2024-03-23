from . import *
import re

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in grads.keys():
            params[key] -= self.lr * grads[key] 

class SGD_Momentum:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay = 0):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        self.pattern = re.compile("Conv_W_.*")
        self.weight_decay = weight_decay

    def update(self, params, grads):
        if self.v is None:
            self.v = {key: mypy.zeros_like(val) for key, val in grads.items()}

        for key, grad in grads.items():
            if self.weight_decay != 0 and self.pattern.match(key):
                grad += self.weight_decay * params[key]

            self.v[key] = self.momentum * self.v[key] + grad
            params[key] -= self.v[key] * self.lr

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key in grads.keys():
                self.h[key] = mypy.zeros_like(params[key])
            
        for key in grads.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (mypy.sqrt(self.h[key]) + 1e-7)