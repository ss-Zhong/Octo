# coding: utf-8
from . import *

# 随机舍入round
def round(M):
    M_int = mypy.floor(M + mypy.random.random(M.shape))
    M_int = M_int.astype(int)
    return M_int


class ConvQuantizer:
    def __init__(self, num_bits = 2):
        # n位量化
        self.n = num_bits

        self.X_scale = 1.
        self.X_zeroPoint = 0.

        self.W_scale = 1.
        self.W_zeroPoint = 0.

        self.dout_scale = 1.
        self.dout_zeroPoint = 0.

    def hintQuant(self):
        hint("X", self.X_scale, self.X_zeroPoint)
        hint("W", self.W_scale, self.W_zeroPoint)
        hint("dout", self.dout_scale, self.dout_zeroPoint)
        pass

    # 计算Scale, Zero Point
    def calcSZSymm(self, tensor_min, tensor_max):
        scale = max(abs(tensor_max), abs(tensor_min)) / ((1 << self.n - 1) - 1)
        zeroPoint = 0.0

        if scale == 0.0:
            scale = 1.0

        return scale, zeroPoint
    
    def calcSZAsymm(self, tensor_min, tensor_max):
        qmax = (1 << self.n - 1) - 1
        scale = (tensor_max - tensor_min) / ((1 << self.n) - 2)
        if scale == 0.0:
            scale = 1.0
            zeroPoint = 0.0
        else:
            zeroPoint = qmax - tensor_max / scale

        return scale, zeroPoint
    
    # 由于X向量一般都是大于0的 单独计算
    def calcSZAsymmT(self, tensor_min, tensor_max):
        scale = (tensor_max - tensor_min) / ((1 << self.n) - 2)
        if scale == 0.0:
            scale = 1.0
            zeroPoint = 0.0
        else:
            zeroPoint = int(0.0 - tensor_min / scale)

        return scale, zeroPoint
    
    # 对称量化X, W
    def quantizeXSymm(self, X_f):
        self.X_scale, self.X_zeroPoint = self.calcSZSymm(X_f.min(), X_f.max())
        X_int8 = round(X_f / self.X_scale + self.X_zeroPoint)
        return X_int8

    def quantizeWSymm(self, W_f):
        self.W_scale, self.W_zeroPoint = self.calcSZSymm(W_f.min(), W_f.max())
        W_int8 = round(W_f / self.W_scale + self.X_zeroPoint)
        return W_int8
    
    def quantizeDOUTSymm(self, out_f):
        self.dout_scale, self.dout_zeroPoint = self.calcSZSymm(out_f.min(), out_f.max())
        dout_int8 = round(out_f / self.dout_scale + self.dout_zeroPoint)
        return dout_int8
    
    # 非对称量化X, W
    def quantizeXAsymm(self, X_f):
        if X_f.min() * X_f.max()  >= 0.0:
            self.X_scale, self.X_zeroPoint = self.calcSZAsymmT(X_f.min(), X_f.max())
        else:
            self.X_scale, self.X_zeroPoint = self.calcSZAsymm(X_f.min(), X_f.max())
        X_int8 = round(X_f / self.X_scale + self.X_zeroPoint)
        return X_int8

    def quantizeWAsymm(self, W_f):
        if W_f.min() * W_f.max()  >= 0.0:
            self.W_scale, self.W_zeroPoint = self.calcSZAsymmT(W_f.min(), W_f.max())
        else:
            self.W_scale, self.W_zeroPoint = self.calcSZAsymm(W_f.min(), W_f.max())
        W_int8 = round(W_f / self.W_scale + self.W_zeroPoint)
        return W_int8
    
    def compensationAsymm(self, W_fp32, X_fp32, W_int8, X_int8):
        delta_W, delta_X = W_fp32 / self.W_scale - W_int8, X_fp32 / self.X_scale - X_int8
        return mypy.dot(X_fp32 / self.X_scale, delta_W) + mypy.dot(delta_X, W_int8)

    # 去量化  
    def dequantizeX(self, X_int8):
        X_q = (X_int8 - self.X_zeroPoint) * self.X_scale
        return X_q

    def dequantizeW(self, W_int8):
        W_q = (W_int8 - self.W_zeroPoint) * self.W_scale
        return W_q
    
    def dequantizeY(self, Y_int):
        return Y_int * self.W_scale * self.X_scale
    
    def dequantize_dW(self, dW_int):
        dW_q = dW_int * self.X_scale * self.dout_scale
        return dW_q
    
    def dequantize_dcol(self, dcol_int):
        dcol_q = dcol_int * self.W_scale * self.dout_scale
        return dcol_q