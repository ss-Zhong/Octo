from libs.datasets_lib import cifar10
from libs.ml_lib import optimizer
from models.vgg11 import *
from libs.ml_lib.trainer import *
from datetime import datetime
import time

'''
Parameter Setting
'''
# experiment setting
train_mode = False
date_time = datetime.now().strftime("%Y%m%d_%H%M")
csvPath = 'result/data/' + date_time + '.csv'
saveModelFile = 'result/model/' + date_time + '.pkl'
loadModelFile = "result/model/20240114_1619.pkl"
# loadModelFile = None

# train setting
batch_size = 100
epoch_num = 1
lr = 0.002
quant_mode = QuantMode.FullPrecision

# model setting
class_num = 10
model = VGG11(optimizer=optimizer.SGD(lr=lr), 
              quant_mode = quant_mode, 
              class_num = class_num, 
              load_file = loadModelFile)
dataset = cifar10.loadCIFAR10(normalize=True)


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
    
    # 测试模型
    else:
        trainer = Trainer(model=model, dataset=dataset, class_num=class_num, batch_size=batch_size, epoch_num=epoch_num)
        
        print("acc: ", trainer.accuracy(batch_size))