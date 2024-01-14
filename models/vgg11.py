from libs.ml_lib.layers import *
import pickle

cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

class VGG11:
    def __init__(self, optimizer, quant_mode=QuantMode.FullPrecision, class_num=10, load_file = None):
        self.optimizer = optimizer
        self.quant_mode = quant_mode
        self.params = {}
        self.layers = self.make_layers(cfg, input_channel = 3, batch_norm=True, class_num = class_num, file_name = load_file)
        self.softmax = SoftmaxWithLoss()

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
        grads = {}
        i = len(self.layers)

        for layer in reversed(self.layers):
            # 反向传播
            dout = layer.backward(dout)

            # 记录grads
            i -= 1
            if isinstance(layer, Conv):
                grads['Conv_W_' + str(i)] = layer.dW
                grads['Conv_b_' + str(i)] = layer.db

            elif isinstance(layer, BatchNorm):
                grads['BN_Gamma_' + str(i)] = layer.dgamma
                grads['BN_Beta_' + str(i)] = layer.dbeta

            elif isinstance(layer, FC):
                grads['FC_W_' + str(i)] = layer.dW
                grads['FC_b_' + str(i)] = layer.db

        self.optimizer.update(self.params, grads)

        return dout

    # batch_norm 从一个隐藏层获取输出并在将它们作为下一个隐藏层的输入传递之前对其进行"标准化"
    def make_layers(self, cfg, batch_norm=False, input_channel = 3, class_num = 10, file_name = None):
        layers = []

        # 有预训练模型
        if file_name:
            hint("=======        Load Params        =======")
            with open(file_name, 'rb') as f:
                self.params = pickle.load(f)

            for l in cfg:
                if l == 'M':
                    layers += [Pooling(2, 2)]
                    continue

                layers += [Conv(W=self.params['Conv_W_' + str(len(layers))],
                                b=self.params['Conv_b_' + str(len(layers))],
                                pad=1, 
                                quant_mode=self.quant_mode)]

                if batch_norm == True:
                    # print('BN_running_mean_1', self.params['BN_running_mean_1'])
                    # print('BN_running_var_1', self.params['BN_running_var_1'])
                    # input()
                    layers += [BatchNorm(
                        gamma=self.params['BN_Gamma_' + str(len(layers))],
                        beta=self.params['BN_Beta_' + str(len(layers))],
                        running_mean=self.params['BN_running_mean_' + str(len(layers))],
                        running_var=self.params['BN_running_var_' + str(len(layers))],
                        epsilon=self.params['BN_Epsilon_' + str(len(layers))]
                    )]

                layers += [Relu()]
                input_channel = l

            layers += [FC(self.params['FC_W_' + str(len(layers))], self.params['FC_b_' + str(len(layers))]), 
                       FC(self.params['FC_W_' + str(len(layers)+1)], self.params['FC_b_' + str(len(layers)+1)]),
                       FC(self.params['FC_W_' + str(len(layers)+2)], self.params['FC_b_' + str(len(layers)+2)])]
            
            hint("=======    Success Load Params    =======")
        
        # 无预训练模型
        else:
            hint("=======        Init Params        =======")

            for l in cfg:
                if l == 'M':
                    layers += [Pooling(2, 2)]
                    continue

                self.params['Conv_W_' + str(len(layers))] = mypy.random.normal(loc=0, scale=0.01, size=(l, input_channel, 3, 3))
                self.params['Conv_b_' + str(len(layers))] = mypy.zeros(l, dtype=float)
                layers += [Conv(W=self.params['Conv_W_' + str(len(layers))],
                                b=self.params['Conv_b_' + str(len(layers))],
                                pad=1,
                                quant_mode=self.quant_mode)]
                if batch_norm == True:
                    self.params['BN_Gamma_' + str(len(layers))] = mypy.ones((1, l, 1, 1))
                    self.params['BN_Beta_' + str(len(layers))] = mypy.zeros((1, l, 1, 1))
                    self.params['BN_running_mean_' + str(len(layers))] = mypy.zeros((1, l, 1, 1))
                    self.params['BN_running_var_' + str(len(layers))] = mypy.zeros((1, l, 1, 1))
                    self.params['BN_Epsilon_' + str(len(layers))] = 1e-5
                    layers += [BatchNorm(
                        gamma=self.params['BN_Gamma_' + str(len(layers))],
                        beta=self.params['BN_Beta_' + str(len(layers))],
                        running_mean=self.params['BN_running_mean_' + str(len(layers))],
                        running_var=self.params['BN_running_var_' + str(len(layers))],
                        epsilon=self.params['BN_Epsilon_' + str(len(layers))]
                    )]
                layers += [Relu()]
                input_channel = l
            
            FC_channel = [512, 4096, 4096, class_num]
            for i in range(len(FC_channel) - 1):
                FC_in = FC_channel[i]
                FC_out = FC_channel[i+1]
                self.params['FC_W_' + str(len(layers))] = mypy.random.normal(loc=0, scale=mypy.sqrt(2 / FC_in), size=(FC_in, FC_out))
                self.params['FC_b_' + str(len(layers))] = mypy.zeros((1, FC_out))
                layers += [FC(W=self.params['FC_W_' + str(len(layers))], 
                              b=self.params['FC_b_' + str(len(layers))])]
                
            hint("=======    Success Init Params    =======")
        
        return layers
    
    def save_params(self, file_name="result/model/vgg11.pkl"):
        hint("=======        Save Params        =======")

        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchNorm):
                self.params['BN_running_mean_' + str(i)] = layer.running_mean
                self.params['BN_running_var_' + str(i)] = layer.running_var

        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

        hint("=======    Success Save Params    =======")