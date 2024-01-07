from libs.ml_lib.layers import *
from libs.ml_lib import optimizer

cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

class VGG11:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.train_mode = True
        self.layers = self.make_layers(cfg, input_channel = 3, batch_norm=True)
        self.Softmax = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                x = layer.forward(x, False)
            else:
                x = layer.forward(x)

        res, _ = self.Softmax.forward(x)
        return res
        
    def forward(self, x, t):
        for layer in self.layers:
            x = layer.forward(x)

        _, loss = self.Softmax.forward(x, t)
        return loss
    
    def backward(self):
        dout = self.Softmax.backward()

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    # batch_norm 从一个隐藏层获取输出并在将它们作为下一个隐藏层的输入传递之前对其进行"标准化"
    def make_layers(self, cfg, batch_norm=False, input_channel = 3):
        layers = []   
        for l in cfg:
            if l == 'M':
                layers += [Pooling(2, 2)]
                continue
            layers += [Conv(optimizer=self.optimizer, input_channel=input_channel, output_channel=l, kernel_size=3, pad=1)]
            if batch_norm == True:
                layers += [BatchNorm(optimizer=self.optimizer, input_size=l)]
            layers += [Relu()]
            input_channel = l

        layers += [FC(512, 4096, optimizer=self.optimizer), 
                   FC(4096, 4096, optimizer=self.optimizer), 
                   FC(4096, 1000, optimizer=self.optimizer)]
        
        # for i in layers:
        #     print(i)
        # input()
        
        return layers