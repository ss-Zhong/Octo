from libs.datasets_lib import cifar10
from libs.ml_lib import optimizer
from models.vgg11 import *
from libs.ml_lib.trainer import *
from datetime import datetime

'''
Parameter Setting
'''
# experiment setting
train_mode = True
date_time = datetime.now().strftime("%Y%m%d_%H%M")
csvPath = 'result/data/' + date_time + '.csv'
saveModelFile = 'result/model/' + date_time + '.pkl'
loadModelFile = "result/save/vgg-symm-cifar-20-0.01-20240120_1141.pkl"
# loadModelFile = None

# train setting
batch_size = 100
epoch_num = 20
lr = 0.01
# weight_decay = 0

# OCTO setting
'''
quant_mode: Quantization Mode
lac: Loss-aware Compensation
prc: Parameterized Range Clipping
'''
quant_mode = QuantMode.SymmQuantization
lac = False
prc = False

# model setting
dataset = cifar10.loadCIFAR10(normalize=True)
class_num = 10
input_dim = [3, 32]
model = VGG11(optimizer=optimizer.SGD_Momentum(lr=lr), 
              input_dim = input_dim,
              fc1w = 1, # 到达第一个全连接层时图像长宽，CIFAR10为4，MINIST为2
              class_num = class_num, 
              load_file = loadModelFile,
              lac = lac,
              prc = prc)

'''
Train
'''
if __name__ == '__main__':
    # 训练模式
    if train_mode:
        # write csv
        csvFile = open(csvPath, mode='w', newline='')
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['Epoch', 'Batch', 'Loss', 'Accuracy'])

        # 开始计时
        start = datetime.now()

        # train begin
        trainer = Trainer(model=model, dataset=dataset, class_num=class_num,
                          batch_size=batch_size, epoch_num=epoch_num,
                          csvWriter = csvWriter,
                          save_file = saveModelFile)
        trainer.train()

        # 结束计时
        end = datetime.now() 
        print("Time Cost: ", end-start)
        csvWriter.writerow(['time', end-start, '', ''])
        csvFile.close()
    
    # 测试模式
    else:
        trainer = Trainer(model=model, dataset=dataset, class_num=class_num, batch_size=batch_size, epoch_num=epoch_num)
        print("acc: ", trainer.accuracy(batch_size))