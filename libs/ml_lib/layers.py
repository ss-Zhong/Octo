import numpy as np
from libs.ml_lib.utils import *
from libs.ml_lib.optimizer import *
from libs.ml_lib.functions import *
from ..quant_lib.quantMode import QuantMode
from ..quant_lib.quantizer import ConvQuantizer
from . import hint


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
        out = 1 / (1 + np.exp(-x))
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
            dx[np.arange(batch_size), self.t] -= 1
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
            # self.W = self.initialize_weights(output_channel, input_channel, kernel_size)
            self.W = np.zeros([output_channel, input_channel, kernel_size, kernel_size], dtype=float)
            self.b = np.zeros(output_channel, dtype=float)
        else:
            print("卷积核未初始化")

        self.stride = stride
        self.pad = pad
        self.optimizer = optimizer

        self.x = None   
        self.col = None
        self.col_W = None

        self.quantizer = ConvQuantizer()
        self.quant_mode = quant_mode
    
    def initialize_weights(self, output_channel, input_channel, kernel_size):
        n_in = input_channel * kernel_size * kernel_size
        n_out = output_channel * kernel_size * kernel_size
        a = np.sqrt(6 / (n_in + n_out))
        return np.random.uniform(low=-a, high=a, size=(output_channel, input_channel, kernel_size, kernel_size))

    def forward(self, x):
        # 全精度，不进行量化
        if self.quant_mode == QuantMode.FullPrecision:
            N_W, _, H_W, W_W = self.W.shape
            N, _, H, W = x.shape

            out_h = (H + 2*self.pad - H_W)//self.stride + 1
            out_w = (W + 2*self.pad - W_W)//self.stride + 1

            col = im2col(x, H_W, W_W, self.stride, self.pad)                # col: N*out_h*out_w, C*H_W*W_W
            col_W = self.W.reshape(N_W, -1).T                               # col_W: C*H_W*W_W, N_W

            out = np.dot(col, col_W) + self.b                               # out: N*out_h*out_w, N_W b: N_W
            out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)    # out: N, N_W, out_h, out_w

            self.x = x
            self.col = col
            self.col_W = col_W

            return out

        elif self.quant_mode == QuantMode.Quantization:
            pass

    def backward(self, dout):
        # 全精度，不进行量化
        if self.quant_mode == QuantMode.FullPrecision:
            N_W, C, H_W, W_W = self.W.shape

            # dout: (N, N_W, out_h, out_w) -> (N*out_h*out_w, N_W)
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, N_W)

            db = np.sum(dout, axis=0)
            dW = np.dot(dout.T, self.col).reshape(N_W, C, H_W, W_W)
            # print(dW)
            # input()

            # self.b -= self.lr * db
            # self.W -= self.lr * dW
            self.optimizer.update(self.W, dW)
            self.optimizer.update(self.b, db)

            dcol = np.dot(dout, self.col_W.T)
            dx = col2im(dcol, self.x.shape, H_W, W_W, self.stride, self.pad)

            return dx


class FC:
    def __init__(self, input_channel, output_channel, optimizer=SGD()):
        self.W = self.initialize_weights(input_channel, output_channel)
        self.b = np.zeros((1, output_channel))
        self.optimizer = optimizer

        self.x = None

    def initialize_weights(self, input_channel, output_channel):
        a = np.sqrt(2 / input_channel)
        return np.random.normal(loc=0, scale=a, size=(input_channel, output_channel))

    def forward(self, x):
        self.x = np.squeeze(x)
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.W.T)

        self.optimizer.update(self.W, dW)
        self.optimizer.update(self.b, db)

        return dx


class BatchNorm:
    def __init__(self, input_size, optimizer, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.input_size = input_size
        self.gamma = np.ones((1, input_size, 1, 1))  # 初始化为1
        self.beta = np.zeros((1, input_size, 1, 1))  # 初始化为0
        self.running_mean = np.zeros((1, input_size, 1, 1))
        self.running_var = np.zeros((1, input_size, 1, 1))

        self.optimizer = optimizer

    def forward(self, x, train_mode = True):
        print(train_mode)
        if train_mode:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

            # 更新 running mean 和 running var
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout):
        batch_size = dout.shape[0]

        dx_normalized = dout * self.gamma
        dvar = np.sum(dx_normalized * (self.x - self.mean) * -0.5 * (self.var + self.epsilon) ** (-1.5), axis=0)
        dmean = np.sum(dx_normalized * -1 / np.sqrt(self.var + self.epsilon), axis=0) + dvar * np.sum(-2 * (self.x - self.mean), axis=0) / batch_size
        dx = dx_normalized / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.x - self.mean) / batch_size + dmean / batch_size

        dgamma = np.sum(dout * self.x_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)

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
            arg_max = np.argmax(col, axis=1)
            out = np.max(col, axis=1)
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
        dmax = np.zeros((dout.size, pool_size))                                     # dmax: dout.size * pool_size
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten() # dmax相应位置变为梯度

        # dmax = dmax.reshape(dout.shape + (pool_size,))                              # dmax: N out_h out_w C pool_size
        # dmax = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)      # dcol: N*out_h*out_w C*pool_size
        dx = col2im(dmax, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx