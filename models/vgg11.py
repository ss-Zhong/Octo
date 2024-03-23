from libs.ml_lib.layers import *
import pickle
import time

"""
    [Vgg11]

    Conv - (LAC) - BatchNorm - Relu - Pooling
    Conv - (LAC) - BatchNorm - Relu - Pooling
    Conv - (LAC) - BatchNorm - Relu
    Conv - (LAC) - BatchNorm - Relu - Pooling
    Conv - (LAC) - BatchNorm - Relu
    Conv - (LAC) - BatchNorm - Relu - Pooling
    Conv - (LAC) - BatchNorm - Relu
    Conv - (LAC) - BatchNorm - Relu - Pooling
    FC - Relu - Dropout
    FC - Relu - Dropout
    FC
    Softmax
"""
cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

class VGG11:
    def __init__(self, optimizer, quant_mode=QuantMode.FullPrecision, input_dim = [3, 32], batch_norm = True, class_num=10, load_file = None, lac = False, prc = False):
        self.optimizer = optimizer
        self.quant_mode = quant_mode
        self.params = {}
        self.layers = self.make_layers(cfg, input_dim = input_dim, batch_norm=batch_norm, class_num=class_num, file_name=load_file, lac=lac, prc=prc)
        self.softmax = SoftmaxWithLoss()

    def predict(self, x, t = None):
        for layer in self.layers:
            if isinstance(layer, BatchNorm) or isinstance(layer, Dropout):
                x = layer.forward(x, False)
            else:
                x = layer.forward(x)

        res, loss = self.softmax.forward(x, t)
        return res, loss
        
    def forward(self, x, t, log = False):
        for layer in self.layers:
            x = layer.forward(x)
        res, loss = self.softmax.forward(x, t)

        res = mypy.argmax(res, axis=1)
        t = mypy.argmax(t, axis=1)
        acc = mypy.sum(res == t)
        return acc, loss
    
    def backward(self, log = False):
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

            elif isinstance(layer, Compensation):
                grads['LAC_Beta_' + str(i)] = layer.d_beta
                

        self.optimizer.update(self.params, grads)
        return dout

    def make_layers(self, cfg, batch_norm=False, input_dim = [3, 32], class_num = 10, file_name = None, lac = False, prc = False):
        layers = []
        make_way = 'Init'

        # 有预训练模型，则加载预训练模型
        if file_name:
            make_way = 'Load'
            with open(file_name, 'rb') as f:
                self.params = pickle.load(f)

        hint(f"=======        {make_way} Params        =======")

        for l in cfg:
            if l == 'M':
                layers += [Pooling(2, 2)]
                input_dim[1] //= 2
                continue

            # 卷积层
            if file_name is None:
                self.params['Conv_W_' + str(len(layers))] = Conv.initialize_weights(l, (input_dim[0], 3, 3))
                self.params['Conv_b_' + str(len(layers))] = Conv.initialize_bias(l, 1)    

            if input_dim[1] == 14:
                layers += [Conv(W=self.params['Conv_W_' + str(len(layers))],
                                b=self.params['Conv_b_' + str(len(layers))],
                                pad=2,
                                quant_mode=self.quant_mode)]
                input_dim[1] += 2
            else:
                layers += [Conv(W=self.params['Conv_W_' + str(len(layers))],
                                b=self.params['Conv_b_' + str(len(layers))],
                                pad=1,
                                quant_mode=self.quant_mode,
                                prc =prc)]

            # Loss-aware Compensation
            if lac:
                if file_name is None:
                    self.params['LAC_Beta_' + str(len(layers))] = mypy.random.normal(0.0, 1.0, size=[l, input_dim[1], input_dim[1]])
                layers += [Compensation(quantizer = layers[-1].quantizer,
                                        beta=self.params['LAC_Beta_' + str(len(layers))])]

            # Batch Normalization
            if batch_norm == True:
                if file_name is None:
                    if lac:
                        self.params['BN_Gamma_' + str(len(layers))] = mypy.ones((1, l, input_dim[1], input_dim[1]))
                        self.params['BN_Beta_' + str(len(layers))] = mypy.zeros((1, l, input_dim[1], input_dim[1]))
                        self.params['BN_running_mean_' + str(len(layers))] = mypy.zeros((1, l, input_dim[1], input_dim[1]))
                        self.params['BN_running_var_' + str(len(layers))] = mypy.zeros((1, l, input_dim[1], input_dim[1]))
                    else:
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
            input_dim[0] = l

        # 全连接层
        if file_name:
            layers += [FC(self.params['FC_W_' + str(len(layers))], self.params['FC_b_' + str(len(layers))]),
                       Relu(), Dropout(0.5),
                       FC(self.params['FC_W_' + str(len(layers)+3)], self.params['FC_b_' + str(len(layers)+3)]),
                       Relu(), Dropout(0.5),
                       FC(self.params['FC_W_' + str(len(layers)+6)], self.params['FC_b_' + str(len(layers)+6)])]

        else:
            FC_channel = [512, 4096, 4096, class_num]
            for i in range(len(FC_channel) - 1):
                FC_in = int(FC_channel[i])
                FC_out = FC_channel[i+1]
                self.params['FC_W_' + str(len(layers))] = FC.initialize_weights(FC_out, FC_in)
                self.params['FC_b_' + str(len(layers))] = FC.initialize_bias(FC_out, FC_in)
                layers += [FC(W=self.params['FC_W_' + str(len(layers))], 
                                b=self.params['FC_b_' + str(len(layers))])]
                
                if i < 2:
                    layers += [Relu(), Dropout(0.5)]
            
        hint(f"=======    Success {make_way} Params    =======")
    
        return layers

    def save_params(self, file_name = None):
        hint("=======        Save Params        =======")

        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchNorm):
                self.params['BN_running_mean_' + str(i)] = layer.running_mean
                self.params['BN_running_var_' + str(i)] = layer.running_var

        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

        hint("=======    Success Save Params    =======")