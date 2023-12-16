import numpy as np

# 随机舍入round
def round(M):
    M_q = np.floor(M + np.random.random(M.shape))
    M_q = M_q.astype(np.int)
    return M_q

class ConvQuantizer:

    def __init__(self, num_bits = 8):
        self.n = num_bits

        self.X_scale = 1.
        self.X_zeroPoint = 0.

        self.W_scale = 1.
        self.W_zeroPoint = 0.

    # 计算Scale, Zero Point
    def calcSZ(self, min, max):
        scale = (max - min) / ((1 << self.n) - 1)
        zeroPoint = int(0. - min / scale)
        return scale, zeroPoint
    
    # 量化X, W
    def quantizeX(self, X_f):
        self.X_scale, self.X_zeroPoint = self.calcSZ(X_f.min(), X_f.max())
        X_q = round(X_f / self.X_scale + self.X_zeroPoint)
        return X_q

    def quantizeW(self, W_f):
        self.W_scale, self.W_zero_point = self.calcSZ(W_f.min(), W_f.max())
        W_q = round(W_f / self.W_scale + self.X_zeroPoint)
        return W_q

    # 去量化X, W
    def dequantizeX(self, X_q):
        X_f = (X_q - self.X_zeroPoint) * self.X_scale
        return X_f

    def dequantizeW(self, W_q):
        W_f = (W_q - self.W_zeroPoint) * self.W_scale
        return W_f
    
    def dequantizeY(self, Y_q):
        return Y_q * self.X_scale * self.W_scale