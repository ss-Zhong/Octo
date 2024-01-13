from libs.ml_lib.optimizer import *
from libs.ml_lib.utils import *
from libs.ml_lib.functions import *
from .. import matrix
from ..quant_lib.quantMode import QuantMode
from ..quant_lib.quantizer import ConvQuantizer
from . import *
import time


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dx = dout
        dx[self.mask] = 0
        # print(dx, '\nrelu')
        # input()
        return dx
    

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
    def __init__(self, W = None, b = None,
                 input_channel = None, output_channel = None, kernel_size=3,
                 stride=1, pad=0, optimizer=SGD(),
                 quant_mode=QuantMode.FullPrecision):
        if W and b: 
            self.W = W
            self.b = b
        elif input_channel and output_channel: 
            self.W = mypy.random.normal(loc=0, scale=0.01, size=(output_channel, input_channel, kernel_size, kernel_size))
            # self.W = mypy.zeros([output_channel, input_channel, kernel_size, kernel_size], dtype=float)
            self.b = mypy.zeros(output_channel, dtype=float)
        else:
            print("卷积核未初始化")

        self.stride = stride
        self.pad = pad
        self.optimizer = optimizer

        self.x = None   
        self.col = None
        self.col_W = None

        self.quantizer = ConvQuantizer(num_bits=8)
        self.quant_mode = quant_mode

    def forward(self, x):
        N_W, _, H_W, W_W = self.W.shape
        N, _, H, W = x.shape

        out_h = (H + 2*self.pad - H_W)//self.stride + 1
        out_w = (W + 2*self.pad - W_W)//self.stride + 1

        # 全精度，不进行量化
        if self.quant_mode == QuantMode.FullPrecision:
            col = im2col(x, H_W, W_W, self.stride, self.pad)                # col: N*out_h*out_w, C*H_W*W_W
            col_W = self.W.reshape(N_W, -1).T                               # col_W: C*H_W*W_W, N_W

            out = mypy.dot(col, col_W) + self.b                               # out: N*out_h*out_w, N_W b: N_W
            out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)    # out: N, N_W, out_h, out_w

            self.x = x
            self.col = col
            self.col_W = col_W

        # 对称量化
        elif self.quant_mode == QuantMode.SymmQuantization:
            x_int8 = self.quantizer.quantizeXSymm(x)
            W_int8 = self.quantizer.quantizeWSymm(self.W)

            col_int8 = im2col(x_int8, H_W, W_W, self.stride, self.pad)
            col_W_int8 = W_int8.reshape(N_W, -1).T

            # 调用pybind+Eigen
            out_int8 = mypy.dot(col_int8, col_W_int8) #
            out = self.quantizer.dequantizeY(out_int8) + self.b
            out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

            self.x = self.quantizer.dequantizeX(x_int8)
            # 优化??????
            self.col = self.quantizer.dequantizeX(col_int8)
            self.col_W = self.quantizer.dequantizeW(col_W_int8)

        return out

    def backward(self, dout):
        N_W, C, H_W, W_W = self.W.shape
        # dout: (N, N_W, out_h, out_w) -> (N*out_h*out_w, N_W)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, N_W)
        db = mypy.sum(dout, axis=0)

        # 全精度，不进行量化
        if self.quant_mode == QuantMode.FullPrecision:
            dW = mypy.dot(dout.T, self.col).reshape(N_W, C, H_W, W_W)

            self.optimizer.update(self.W, dW)
            self.optimizer.update(self.b, db)

            dcol = mypy.dot(dout, self.col_W.T)
            dx = col2im(dcol, self.x.shape, H_W, W_W, self.stride, self.pad)
        
        # 对称量化
        elif self.quant_mode == QuantMode.SymmQuantization:
            col_int8 = self.quantizer.quantizeXSymm(self.col)
            col_W_int8 = self.quantizer.quantizeWSymm(self.col_W)
            dout_int8 = self.quantizer.quantizeDOUTSymm(dout)

            dW_int = mypy.dot(dout_int8.T, col_int8).reshape(N_W, C, H_W, W_W) #
            dW_q = self.quantizer.dequantize_dW(dW_int)
            self.optimizer.update(self.W, dW_q)
            self.optimizer.update(self.b, db)

            dcol_int = mypy.dot(dout_int8, col_W_int8.T) #
            dcol_q = self.quantizer.dequantize_dcol(dcol_int)
            dx = col2im(dcol_q, self.x.shape, H_W, W_W, self.stride, self.pad)

        return dx


class FC:
    def __init__(self, input_channel, output_channel, optimizer=SGD()):
        self.W = self.initialize_weights(input_channel, output_channel)
        self.b = mypy.zeros((1, output_channel))
        self.optimizer = optimizer

        self.x = None

    def initialize_weights(self, input_channel, output_channel):
        a = mypy.sqrt(2 / input_channel)
        return mypy.random.normal(loc=0, scale=a, size=(input_channel, output_channel))

    def forward(self, x):
        self.x = mypy.squeeze(x)
        out = mypy.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dW = mypy.dot(self.x.T, dout)
        db = mypy.sum(dout, axis=0, keepdims=True)
        dx = mypy.dot(dout, self.W.T)

        self.optimizer.update(self.W, dW)
        self.optimizer.update(self.b, db)

        return dx
    

class BatchNorm:
    def __init__(self, input_size, optimizer, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.input_size = input_size
        self.gamma = mypy.ones((1, input_size, 1, 1))  # 初始化为1
        self.beta = mypy.zeros((1, input_size, 1, 1))  # 初始化为0
        self.running_mean = mypy.zeros((1, input_size, 1, 1))
        self.running_var = mypy.zeros((1, input_size, 1, 1))
        self.optimizer = optimizer

        self.x = None
        self.mean = None
        self.var = None
        self.x_normalized = None

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

        dgamma = mypy.sum(dout * self.x_normalized, axis=(0, 2, 3), keepdims=True)
        dbeta = mypy.sum(dout, axis=(0, 2, 3), keepdims=True)
        dx_normalized = dout * self.gamma

        dvar = mypy.sum(dx_normalized * (self.x - self.mean) * -0.5 * (self.var + self.epsilon) ** (-1.5), axis=0)
        dmean = mypy.sum(dx_normalized * -1 / mypy.sqrt(self.var + self.epsilon), axis=0) + dvar * mypy.sum(-2 * (self.x - self.mean), axis=0) / batch_size
        dx = dx_normalized / mypy.sqrt(self.var + self.epsilon) + dvar * 2 * (self.x - self.mean) / batch_size + dmean / batch_size

        # 更新 gamma 和 beta
        self.optimizer.update(self.gamma, dgamma)
        self.optimizer.update(self.beta, dbeta)

        return dx


class Pooling:
    def __init__(self, pool_w, pool_h, stride = None, pad = 0, pool_type = 0):
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

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        out = None

        # MaxPooling
        if self.pool_type == 0:
            arg_max = mypy.argmax(col, axis=1)
            out = mypy.max(col, axis=1)
            out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2) # out: N C out_h out_w

            self.x = x
            self.arg_max = arg_max
        
        # AvgPooling
        elif self.pool_type == 1:
            hint("AvgPooling unfinished")
            pass

        else:
            print("[ERROR] NoneType of Pooling")

        return out
    
    def backward(self, dout):
        if dout.ndim == 2:
            dout = dout.reshape(dout.shape[0], dout.shape[1], 1, 1)
        dout = dout.transpose(0, 2, 3, 1)                                           # dout: N out_h out_w C
        
        pool_size = self.pool_h * self.pool_w
        dmax = mypy.zeros((dout.size, pool_size))                                     # dmax: dout.size * pool_size
        dmax[mypy.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten() # dmax相应位置变为梯度

        # dmax = dmax.reshape(dout.shape + (pool_size,))                              # dmax: N out_h out_w C pool_size
        # dmax = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)      # dcol: N*out_h*out_w C*pool_size
        dx = col2im(dmax, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx