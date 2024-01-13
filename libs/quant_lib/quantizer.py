# coding: utf-8
from . import *

# 随机舍入round
def round(M):
    M_int = mypy.floor(M + mypy.random.random(M.shape))
    # M_int = M_int.astype(int)
    return M_int


class ConvQuantizer:
    def __init__(self, num_bits = 8):
        # 8位量化
        self.n = num_bits

        self.X_scale = 1.
        self.X_zeroPoint = 0.

        self.W_scale = 1.
        self.W_zeroPoint = 0.

        self.dout_scale = 1.
        self.dout_zeroPoint = 0.

    # 计算Scale, Zero Point
    def calcSZSymm(self, tensor_min, tensor_max):
        scale = max(abs(tensor_max), abs(tensor_min)) / (1 << (self.n - 1))
        zeroPoint = 0.0

        if scale == 0.0:
            scale = 1.0

        return scale, zeroPoint
    
    def calcSZAsymm(self, tensor_min, tensor_max):
        scale = (tensor_max - tensor_min) / ((1 << self.n) - 1)
        zeroPoint = int(0. - tensor_min / scale)

        if scale == 0.0:
            scale = 1.0
            zero_point = 0.0
        else:
            zeroPoint = int(0. - tensor_min / scale)

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
        self.X_scale, self.X_zeroPoint = self.calcSZAsymm(X_f.min(), X_f.max())
        X_int8 = round(X_f / self.X_scale + self.X_zeroPoint)
        return X_int8

    def quantizeWAsymm(self, W_f):
        self.W_scale, self.W_zero_point = self.calcSZAsymm(W_f.min(), W_f.max())
        W_int8 = round(W_f / self.W_scale + self.W_zeroPoint)
        return W_int8

    # 去量化X, W
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