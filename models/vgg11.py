from libs.ml_lib.layers import *
import pickle

cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

class VGG11:
    def __init__(self, optimizer, quant_mode=QuantMode.FullPrecision, class_num=10, load_file = None):
        self.optimizer = optimizer
        self.quant_mode = quant_mode
        self.layers = self.make_layers(cfg, input_channel = 3, batch_norm=True, class_num = class_num)
        self.softmax = SoftmaxWithLoss()

        self.params = {}
        if load_file:
            self.load_params(load_file)
        else:
            self.init_params()

    def predict(self, x):
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                x = layer.forward(x, False)
            else:
                x = layer.forward(x)

        res, _ = self.softmax.forward(x)
        return res
        
    def forward(self, x, t):
        for layer in self.layers:
            x = layer.forward(x)

        _, loss = self.softmax.forward(x, t)
        return loss
    
    def backward(self):
        dout = self.softmax.backward()

        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        self.optimizer.update(self)

        return dout

    # batch_norm 从一个隐藏层获取输出并在将它们作为下一个隐藏层的输入传递之前对其进行"标准化"
    def make_layers(self, cfg, batch_norm=False, input_channel = 3, class_num = 10):
        layers = []   
        for l in cfg:
            if l == 'M':
                layers += [Pooling(2, 2)]
                continue
            layers += [Conv(optimizer=self.optimizer, 
                            input_channel=input_channel, output_channel=l, 
                            kernel_size=3, pad=1, 
                            quant_mode=self.quant_mode)]
            if batch_norm == True:
                layers += [BatchNorm(optimizer=self.optimizer, input_size=l)]
            layers += [Relu()]
            input_channel = l

        layers += [FC(512, 4096, optimizer=self.optimizer), 
                   FC(4096, 4096, optimizer=self.optimizer), 
                   FC(4096, class_num, optimizer=self.optimizer)]
        
        return layers
    
    def save_params(self, file_name="result/model/vgg11.pkl"):
        hint("=======        Save Params        =======")

        params = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv):
                # print(type(layer))
                params['Conv_W_' + str(i+1)] = layer.W
                params['Conv_b_' + str(i+1)] = layer.b
                # FullPrecision only save W, b
                if self.quant_mode != QuantMode.FullPrecision:
                    pass

            elif isinstance(layer, BatchNorm):
                # print(type(layer))
                params['BN_Epsilon_' + str(i+1)] = layer.epsilon
                params['BN_Gamma_' + str(i+1)] = layer.gamma
                params['BN_Beta_' + str(i+1)] = layer.beta
                params['BN_running_mean_' + str(i+1)] = layer.running_mean
                params['BN_running_var_' + str(i+1)] = layer.running_var

            elif isinstance(layer, FC):
                # print(type(layer))
                params['FC_W_' + str(i+1)] = layer.W
                params['FC_b_' + str(i+1)] = layer.b

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

        hint("=======    Success Save Params    =======")

    def load_params(self, file_name="result/model/vgg11.pkl"):
        hint("=======        Load Params        =======")

        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv):
                layer.W = self.params['Conv_W_' + str(i+1)]
                layer.b = self.params['Conv_b_' + str(i+1)]
                # FullPrecision only save W, b
                if self.quant_mode != QuantMode.FullPrecision:
                    pass

            elif isinstance(layer, BatchNorm):
                layer.epsilon = self.params['BN_Epsilon_' + str(i+1)]
                layer.gamma = self.params['BN_Gamma_' + str(i+1)]
                layer.beta = self.params['BN_Beta_' + str(i+1)]
                layer.running_mean = self.params['BN_running_mean_' + str(i+1)]
                layer.running_var = self.params['BN_running_var_' + str(i+1)]

            elif isinstance(layer, FC):
                layer.W = self.params['FC_W_' + str(i+1)]
                layer.b = self.params['FC_b_' + str(i+1)]
        
        hint("=======    Success Load Params    =======")

    def init_params(self, file_name="result/model/vgg11.pkl"):
        hint("=======        Init Params        =======")

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Conv):
                layer.W = self.params['Conv_W_' + str(i+1)]
                layer.b = self.params['Conv_b_' + str(i+1)]
                # FullPrecision only save W, b
                if self.quant_mode != QuantMode.FullPrecision:
                    pass

            elif isinstance(layer, BatchNorm):
                layer.epsilon = self.params['BN_Epsilon_' + str(i+1)]
                layer.gamma = self.params['BN_Gamma_' + str(i+1)]
                layer.beta = self.params['BN_Beta_' + str(i+1)]
                layer.running_mean = self.params['BN_running_mean_' + str(i+1)]
                layer.running_var = self.params['BN_running_var_' + str(i+1)]

            elif isinstance(layer, FC):
                layer.W = self.params['FC_W_' + str(i+1)]
                layer.b = self.params['FC_b_' + str(i+1)]
        
        hint("=======    Success Init Params    =======")