from libs.ml_lib.utils import *
from libs.ml_lib.functions import *
# from .. import matrix
from ..quant_lib.quantMode import QuantMode
from ..quant_lib.quantizer import ConvQuantizer
from . import *
import time

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        out = mypy.maximum(0, x)
        return out
        
    def backward(self, dout):
        return dout * self.mask


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + mypy.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t = None):
        self.y = softmax(x)
        if t is not None:
            self.t = t
            self.loss = cross_entropy_error(self.y, t)
        return self.y, self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[mypy.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Conv:
    def __init__(self, W, b,
                 stride=1, pad=0,
                 quant_mode=QuantMode.FullPrecision, prc = False):
        self.W = W
        self.b = b

        self.stride = stride
        self.pad = pad

        self.x = None   
        self.col = None
        self.col_W = None

        self.quantizer = ConvQuantizer(num_bits=4)
        self.quant_mode = quant_mode
        self.prc = prc

        self.dW = None
        self.db = None
    
    def parametrized_range_clipping(self, tensor):
        # clip_max = mypy.percentile(tensor, 95)
        # clip_min = mypy.percentile(tensor, 5)
        clip_max = tensor.max()
        clip_min = tensor.min()

        if clip_max * clip_min >=0:
            return mypy.clip(tensor, clip_min, clip_max)
        else:
            rangeMin = -clip_min if -clip_min < clip_max else clip_max
            return mypy.clip(tensor, -rangeMin, rangeMin)
    
    def initialize_weights(out_channels, in_shape):
        # 计算权重初始化的标准差
        a = mypy.asarray(in_shape)
        k = 1 / mypy.prod(a)
        weight_std = mypy.sqrt(k)
        # 从均匀分布中采样权重
        weight = mypy.random.uniform(low=-weight_std, high=weight_std, size=(out_channels, *in_shape))
        return weight
    
    def initialize_bias(out_channels, in_channel):
        # 计算bias初始化的标准差
        bias_std = mypy.sqrt(1 / in_channel)
        bias = mypy.random.uniform(low=-bias_std, high=bias_std, size=out_channels)
        return bias

    def forward(self, x):
        s = time.perf_counter()
        self.x = x

        N_W, _, H_W, W_W = self.W.shape
        N, _, H, W = x.shape

        out_h = (H + 2*self.pad - H_W)//self.stride + 1
        out_w = (W + 2*self.pad - W_W)//self.stride + 1

        col = im2col(x, H_W, W_W, out_h, out_w,self.stride, self.pad)
        col_W = self.W.reshape(N_W, -1).T

        # 全精度，不进行量化
        if self.quant_mode == QuantMode.FullPrecision:
            out = mypy.dot(col, col_W) + self.b                               # out: N*out_h*out_w, N_W b: N_W
            out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)    # out: N, N_W, out_h, out_w
            self.col = col
            self.col_W = col_W

        # 量化
        else:
            # Parameterized Range Clipping
            if self.prc:
                col = self.parametrized_range_clipping(col)
                col_W = self.parametrized_range_clipping(col_W)

            # Asymm or Symm
            if self.quant_mode == QuantMode.SymmQuantization:
                col_int8 = self.quantizer.quantizeXSymm(col)
                col_W_int8 = self.quantizer.quantizeWSymm(col_W)
            else:
                col_int8 = self.quantizer.quantizeXAsymm(col)
                col_W_int8 = self.quantizer.quantizeWAsymm(col_W)

            # 是否内置补偿
            compensation = 0
            if self.quant_mode == QuantMode.AsymmQuantizationWithCompensation:
                compensation = self.quantizer.compensationAsymm(col_W, col, col_W_int8, col_int8)

            out_int8 = mypy.dot(col_int8, col_W_int8) # 调用pybind+Eigen??????
            out = self.quantizer.dequantizeY(out_int8 + compensation) + self.b
            out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

            # 由于反向时还需要量化，此处直接不进行解量化
            # if self.quant_mode != QuantMode.SymmQuantization:
            #     self.col = self.quantizer.dequantizeX(col_int8)
            #     self.col_W = self.quantizer.dequantizeW(col_W_int8)
            # else:
            self.col = col_int8
            self.col_W = col_W_int8

        return out

    def backward(self, dout):
        N_W, C, H_W, W_W = self.W.shape
        # dout: (N, N_W, out_h, out_w) -> (N*out_h*out_w, N_W)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, N_W)
        self.db = mypy.sum(dout, axis=0)

        # 全精度，不进行量化
        if self.quant_mode == QuantMode.FullPrecision:
            self.dW = mypy.dot(dout.T, self.col).reshape(N_W, C, H_W, W_W)
            dcol = mypy.dot(dout, self.col_W.T)
            dx = col2im(dcol, self.x.shape, H_W, W_W, self.stride, self.pad)
        
        # 量化
        else:
            if self.prc:
                dout = self.parametrized_range_clipping(dout)

            # if self.quant_mode != QuantMode.SymmQuantization:
            #     col_int8 = self.quantizer.quantizeXSymm(self.col)
            #     col_W_int8 = self.quantizer.quantizeWSymm(self.col_W)
            # else:
            col_int8 = self.col
            col_W_int8 = self.col_W
            dout_int8 = self.quantizer.quantizeDOUTSymm(dout)

            dW_int = mypy.dot(dout_int8.T, col_int8).reshape(N_W, C, H_W, W_W) #
            self.dW = self.quantizer.dequantize_dW(dW_int)

            dcol_int = mypy.dot(dout_int8, col_W_int8.T) #
            dcol_q = self.quantizer.dequantize_dcol(dcol_int)
            dx = col2im(dcol_q, self.x.shape, H_W, W_W, self.stride, self.pad)
        return dx


class FC:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.xshape = None
        
        self.dW = None
        self.db = None

    def initialize_weights(out_channels, in_channel):
        # 计算权重初始化的标准差
        weight_std = mypy.sqrt(1 / in_channel)
        # 从均匀分布中采样权重
        weight = mypy.random.uniform(low=-weight_std, high=weight_std, size=(in_channel, out_channels))
        return weight
    
    def initialize_bias(out_channels, in_channel):
        # 计算bias初始化的标准差
        bias_std = mypy.sqrt(1 / in_channel)
        bias = mypy.random.uniform(low=-bias_std, high=bias_std, size=out_channels)
        return bias

    def forward(self, x):
        self.xshape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        out = mypy.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = mypy.dot(self.x.T, dout)
        self.db = mypy.sum(dout, axis=0)
        dx = mypy.dot(dout, self.W.T).reshape(self.xshape)

        return dx
    

class BatchNorm:
    def __init__(self, gamma, beta, running_mean, running_var, epsilon=1e-5, momentum=0.9):
        self.momentum = momentum
        self.gamma = gamma
        self.beta = beta
        self.running_mean = running_mean
        self.running_var = running_var
        self.epsilon = epsilon

        self.x = None
        self.mean = None
        self.var = None
        self.x_normalized = None

        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_mode = True):
        self.x = x
        if train_mode:
            self.mean = mypy.mean(x, axis=0)
            self.var = mypy.var(x, axis=0)
            self.x_normalized = (x - self.mean) / mypy.sqrt(self.var + self.epsilon)
            out = self.gamma * self.x_normalized + self.beta

            # 更新 running mean 和 running var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            self.x_normalized = (x - self.running_mean) / mypy.sqrt(self.running_var + self.epsilon)
            out = self.gamma * self.x_normalized + self.beta

        return out

    def backward(self, dout):
        batch_size = dout.shape[0]

        # 计算梯度
        self.dgamma = mypy.sum(dout * self.x_normalized, axis=(0, 2, 3), keepdims=True)
        self.dbeta = mypy.sum(dout, axis=(0, 2, 3), keepdims=True)
        dx_normalized = dout * self.gamma

        dvar = mypy.sum(dx_normalized * (self.x - self.mean) * -0.5 * (self.var + self.epsilon) ** (-1.5), axis=0)
        dmean = mypy.sum(dx_normalized * -1 / mypy.sqrt(self.var + self.epsilon), axis=0) + dvar * mypy.sum(-2 * (self.x - self.mean), axis=0) / batch_size
        dx = dx_normalized / mypy.sqrt(self.var + self.epsilon) + dvar * 2 * (self.x - self.mean) / batch_size + dmean / batch_size
        return dx


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_mode=True):
        if train_mode:
            self.mask = mypy.random.binomial(n=1, p=self.dropout_ratio, size=x.shape)
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Pooling:
    def __init__(self, pool_w, pool_h, stride = None, pad = 0, pool_type = 'Max'):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride if stride else pool_w
        self.pad = pad
        # 0: MaxPooling 1: AvgPooling
        self.pool_type = pool_type

        # x存pooling后的值 arg_max存最大值索引
        self.x = None
        self.arg_max = None
    
    def forward(self, x):
        """
        N: number of sample
        C: channels
        W: width of sample
        H: height of sample
        """
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - self.pool_h)//self.stride + 1
        out_w = (W + 2*self.pad - self.pool_w)//self.stride + 1
        col = im2col(x, self.pool_h, self.pool_w, out_h, out_w,self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        out = None

        # MaxPooling
        if self.pool_type == 'Max':
            arg_max = mypy.argmax(col, axis=1)
            out = mypy.max(col, axis=1)
            out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # out: N C out_h out_w

            self.x = x
            self.arg_max = arg_max
        
        # AvgPooling
        elif self.pool_type == 'Avg':
            hint("AvgPooling unfinished")
            pass

        else:
            hint("[ERROR] NoneType of Pooling")

        return out
    
    def backward(self, dout):
        if dout.ndim == 2:
            dout = dout.reshape(dout.shape[0], dout.shape[1], 1, 1)
        dout = dout.transpose(0, 2, 3, 1)                                           # dout: N out_h out_w C
        
        pool_size = self.pool_h * self.pool_w
        dmax = mypy.eye(4)[self.arg_max] * dout.reshape(-1, 1)
        # dmax = mypy.zeros((dout.size, pool_size))                                     # dmax: dout.size * pool_size
        # dmax[mypy.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten() # dmax相应位置变为梯度

        # dmax = dmax.reshape(dout.shape + (pool_size,))                              # dmax: N out_h out_w C pool_size
        # dmax = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)      # dmax: N*out_h*out_w C*pool_size
        dx = col2im(dmax, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx


class Compensation:
    def __init__(self, quantizer, beta):
        self.beta = beta
        self.d_beta = None
        self.mut = 0
        self.quantizer = quantizer

    def forward(self, x):
        self.mut = self.quantizer.X_scale * self.quantizer.W_scale * self.quantizer.X_zeroPoint
        out = x - self.beta * self.mut
        return out

    def backward(self, dout):
        # 计算梯度
        self.d_beta = mypy.sum(-1 * dout * self.mut, axis=0)
        return dout