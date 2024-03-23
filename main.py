from libs.datasets_lib.dataset_ import *
from libs.ml_lib import optimizer
from libs.ml_lib.trainer import *
from libs.quant_lib.quantMode import *
from models.vgg11 import VGG11
from models.alexnet import AlexNet
from datetime import datetime

'''
Parameter Setting
'''
# experiment setting
train_mode = True
date_time = datetime.now().strftime("%Y%m%d_%H%M")
csvPath = 'result/data/' + date_time + '.csv'
saveModelFile = 'result/model/' + date_time + '.pkl'
loadModelFile = None

# train setting
batch_size = 100
epoch_num = 25
lr = 0.01

# OCTO setting
'''
quant_mode: Quantization Mode
lac: Loss-aware Compensation                                                     
prc: Parameterized Range Clipping
'''
quant_mode = QuantMode.AsymmQuantization
lac = (quant_mode == QuantMode.AsymmQuantization) and False
prc = (quant_mode == QuantMode.AsymmQuantization) and True
weigh_decay = 0

# model setting
dataset_name = dataset_.cifar10
dataset, input_dim, class_num = loadDataset(dataset_name)
model = VGG11(optimizer=optimizer.SGD_Momentum(lr=lr, weight_decay=weigh_decay), 
              quant_mode=quant_mode,
              input_dim=input_dim,
              class_num=class_num, 
              load_file=loadModelFile,
              lac = lac,
              prc = prc)

'''
Train
'''
if __name__ == '__main__':
    # 训练模式
    if train_mode:
        # write csv
        csvWriter = None
        if csvPath:
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
        if csvPath:
            csvWriter.writerow(['time', end-start, '', ''])
            csvFile.close()
    
    # 测试模式
    else:
        trainer = Trainer(model=model, dataset=dataset, class_num=class_num, batch_size=batch_size, epoch_num=epoch_num)
        print("acc: ", trainer.accuracy(batch_size))