# coding: utf-8
import numpy as np
from quant_lib.quantMode import QuantMode
from quant_lib.quantizer import ConvQuantizer


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y
    
    def backward(self, dL):
        dx = dL
        dx[self.mask] = 0
        return dx
    

class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, dL):
        dx = dL * (1.0 - self.y) * self.y
        return dx

class Conv:
    def __init__(self, W, b, 
                 stride=1, padding=0, 
                 quantMode=QuantMode.FullPrecision):
        self.W = W
        self.b = b
        self.stride = stride
        self.padding = padding

        self.x = None   
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

        self.quantizer = ConvQuantizer()
        self.quantMode = quantMode
        
    def forward(self, x):
        if self.quantMode == QuantMode.FullPrecision:
            pass





class Pooling:
    def __init__(self, height, width, stride = 1, padding = 0):
        self.height = height
        self.width = width
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        print(x.shape)
        # n: number of sample
        # c: channels
        n, c, h, w = x.shape